import argparse, os, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
from datasets import load_dataset, ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ---------- Argumentos ----------
def get_args():
    p = argparse.ArgumentParser(
        description="Continual fine-tuning para Claim Detection em PT-BR"
    )
    p.add_argument("--checkpoint", type=str, default="Sami92/XLM-R-Large-ClaimDetection",
                   help="Modelo base (ex.: Sami92/... ou xlm-roberta-base)")
    p.add_argument("--train_file", type=str, required=True, help="Caminho do train.(csv|jsonl)")
    p.add_argument("--valid_file", type=str, required=True, help="Caminho do valid.(csv|jsonl)")
    p.add_argument("--test_file",  type=str, default=None, help="Opcional: test.(csv|jsonl)")
    p.add_argument("--text_col",   type=str, default="text", help="Nome da coluna de texto")
    p.add_argument("--label_col",  type=str, default="label", help="Nome da coluna de rótulo")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--train_bsz",  type=int, default=4)
    p.add_argument("--eval_bsz",   type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--fp16", action="store_true", help="Ativa FP16 se houver GPU compatível")
    p.add_argument("--bf16", action="store_true", help="Ativa BF16 (Ampere+) se compatível")
    p.add_argument("--outdir",     type=str, default="xfact-pt")
    p.add_argument("--early_stopping", type=int, default=3, help="Paciencia (nº aval.) para early stopping")
    p.add_argument("--class_weights", action="store_true", help="Ponderar perda por desbalanceamento")
    # LoRA (opcional)
    p.add_argument("--use_lora",   action="store_true", help="Usar LoRA/PEFT (recomendado p/ GPUs pequenas)")
    p.add_argument("--lora_r",     type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()

# ---------- Utilidades ----------
ID2LABEL = {0: "Non-factual", 1: "Factual"}
LABEL2ID = {"Non-factual": 0, "Factual": 1}

def detect_format(path):
    lower = path.lower()
    if lower.endswith(".csv"): return "csv"
    if lower.endswith(".jsonl") or lower.endswith(".json"): return "json"
    raise ValueError("Formato não suportado (use .csv ou .jsonl)")

def load_splits(args):
    data_files = {"train": args.train_file, "validation": args.valid_file}
    if args.test_file: data_files["test"] = args.test_file
    fmt = detect_format(args.train_file)
    if fmt == "csv":
        ds = load_dataset("csv", data_files=data_files)
    else:
        ds = load_dataset("json", data_files=data_files, field=None)
    return ds

def build_label_mapping(ds, label_col):
    """
    Garante mapeamento 0/1 -> Non-factual/Factual.
    Aceita:
      - 0/1
      - 'Factual'/'Non-factual' (case-insensitive)
      - qualquer string -> tenta mapear por substring 'factual' vs 'non'
    """
    def normalize_label(x):
        v = x[label_col]
        if isinstance(v, (int, np.integer)):  # já é 0/1?
            if v in (0,1): return v
        # string:
        s = str(v).strip().lower()
        if s in ("1","true","factual","fact","fato","verdadeiro"): return 1
        if s in ("0","false","non-factual","nonfactual","nao-factual","não-factual",
                 "fake","boato","falso"): return 0
        # fallback heurístico
        return 1 if "fact" in s and "non" not in s else 0

    # detecta dataset para inferir
    splits = ["train","validation"] + (["test"] if "test" in ds else [])
    for sp in splits:
        if label_col not in ds[sp].column_names:
            raise ValueError(f"Coluna de rótulo '{label_col}' não encontrada em {sp}.")
    return normalize_label

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

# ---------- Trainer com class weights (opcional) ----------
import torch
from typing import Optional, Dict, Any

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor]=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# ---------- Early Stopping Callback ----------
from transformers.trainer_callback import TrainerCallback

class EarlyStoppingSimple(TrainerCallback):
    def __init__(self, patience=3, metric_name="eval_f1", minimize=False):
        self.patience = patience
        self.metric_name = metric_name
        self.best = None
        self.count = 0
        self.minimize = minimize

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        val = metrics.get(self.metric_name)
        if val is None:
            return
        improved = (val < self.best) if (self.best is not None and self.minimize) else (val > (self.best if self.best is not None else -1e9))
        if self.best is None or improved:
            self.best = val
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                control.should_training_stop = True

# ---------- Main ----------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Carrega dataset
    ds = load_splits(args)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint)

    # Normaliza rótulos para 0/1
    normalize_label = build_label_mapping(ds, args.label_col)

    def preprocess(batch):
        texts = batch[args.text_col]
        labels = [normalize_label({"label": l}) for l in batch[args.label_col]]
        enc = tok(texts, truncation=True, max_length=args.max_length)
        enc["labels"] = labels
        return enc

    for split in ["train","validation"] + (["test"] if "test" in ds else []):
        if args.text_col not in ds[split].column_names:
            raise ValueError(f"Coluna de texto '{args.text_col}' não encontrada em {split}.")

    ds = ds.map(preprocess, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in [args.text_col, args.label_col]])

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # LoRA (opcional)
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    # Class weights (opcional)
    class_weights = None
    if args.class_weights:
        # conta rótulos no train
        y = np.array(ds["train"]["labels"])
        n0 = (y == 0).sum()
        n1 = (y == 1).sum()
        # inversamente proporcionais à frequência
        w0 = 1.0 / max(n0, 1)
        w1 = 1.0 / max(n1, 1)
        s = w0 + w1
        class_weights = torch.tensor([w0/s, w1/s], dtype=torch.float32)

    # Args de treino
    targs = TrainingArguments(
        output_dir=args.outdir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    callbacks = [EarlyStoppingSimple(patience=args.early_stopping, metric_name="eval_f1")]

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights
    )

    # Treina
    trainer.train()

    # Avalia
    print("\n# Validação:")
    print(trainer.evaluate())

    if "test" in ds:
        print("\n# Teste:")
        print(trainer.evaluate(ds["test"]))

    # Salva
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)

    # Exporta rótulos para conferência
    with open(os.path.join(args.outdir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {args.outdir}")
    print("Rótulos: ", ID2LABEL)

if __name__ == "__main__":
    main()
