# train_3class.py
# Fine-tuning 3 classes (non-factual, factual-non-important, factual-checkworthy)
# com opção de LoRA/PEFT e class weights.
# Requer: transformers>=4.44, datasets>=2.14, evaluate, (opcional) peft, bitsandbytes, scikit-learn

import os
import argparse
import numpy as np
from typing import Dict, Any, Optional

import torch
from transformers import AutoConfig, BitsAndBytesConfig
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ==== (opcional) LoRA/PEFT ====
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# ============================
# 1) Argumentos
# ============================
def get_args():
    ap = argparse.ArgumentParser(description="Fine-tune XLM-R Large ClaimDetection (3 classes)")

    # Dados
    ap.add_argument("--train_csv", type=str, required=True, help="Caminho do CSV de treino")
    ap.add_argument("--val_csv",   type=str, required=True, help="Caminho do CSV de validação")
    ap.add_argument("--test_csv",  type=str, required=True, help="Caminho do CSV de teste")
    ap.add_argument("--text_col_primary",   type=str, default="text_pt",
                    help="Coluna preferencial do texto (default: text_pt)")
    ap.add_argument("--text_col_fallback",  type=str, default="Text",
                    help="Coluna fallback do texto se a principal estiver vazia (default: Text)")
    ap.add_argument("--label_col", type=str, default="Verdict",
                    help="Coluna de rótulo no CSV (default: Verdict)")

    # Modelo / Tokenizer
    ap.add_argument("--checkpoint", type=str,
                    default="Sami92/XLM-R-Large-ClaimDetection",
                    help="Checkpoint base (default: Sami92/XLM-R-Large-ClaimDetection)")
    ap.add_argument("--output_dir", type=str, default="xlmr_claims_3c_out",
                    help="Pasta de saída (modelo e logs)")

    # Treino
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--train_bs", type=int, default=8, help="Batch por dispositivo (treino)")
    ap.add_argument("--eval_bs",  type=int, default=16, help="Batch por dispositivo (val/test)")
    ap.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", help="Usa FP16")
    ap.add_argument("--cosine", action="store_true", help="Usa scheduler cosine (senão linear)")
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_checkpoint", action="store_true", help="Ativa gradient checkpointing")

    # LoRA / 8-bit
    ap.add_argument("--lora", action="store_true", help="Ativar LoRA/PEFT")
    ap.add_argument("--load_in_8bit", action="store_true", help="Carregar modelo em 8-bit (requer bitsandbytes)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Pesos de classe
    ap.add_argument("--class_weights", choices=["none", "auto"], default="auto",
                    help="none=sem pesos; auto=pesos proporcionais ao inverso da frequência (default: auto)")

    # Diversos
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--no_report_to", action="store_true", help="Não reportar para wandb/hub (report_to=none)")

    return ap.parse_args()


# ============================
# 2) Rótulos e normalização
# ============================
ID2LABEL = {
    0: "non-factual",
    1: "factual-non-important",
    2: "factual-checkworthy"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = 3


def normalize_label(raw: str) -> str:
    """
    Normaliza rótulos heterogêneos para o conjunto canônico:
    {non-factual, factual-non-important, factual-checkworthy}
    Ajuste aqui para cobrir casos do seu dataset.
    """
    if raw is None:
        return "non-factual"
    s = str(raw).strip().lower()

    # não factual
    if s in {"não factual", "nao factual", "non-factual", "nf", "0", "-1", "non factual"}:
        return "non-factual"

    # factual não importante
    if s in {"factual não importante", "factual nao importante",
             "factual-non-important", "fni", "1"}:
        return "factual-non-important"

    # check-worthy factual (padrão)
    # Exemplos: "check-worthy factual statement", "checkworthy", "2"
    return "factual-checkworthy"


# ============================
# 3) Dataset + Tokenização
# ============================
from datasets import DatasetDict, ClassLabel
from transformers import AutoTokenizer

def build_datasets(args) -> Dict[str, Any]:
    def pick_text(ex):
        txt = (ex.get(args.text_col_primary) or "").strip() if args.text_col_primary in ex else ""
        if not txt:
            txt = (ex.get(args.text_col_fallback) or "").strip() if args.text_col_fallback in ex else ""
        ex["text"] = txt
        return ex

    def map_label(ex):
        raw = ex.get(args.label_col, None)
        norm = normalize_label(raw)
        ex["labels"] = LABEL2ID[norm]  # 0/1/2
        return ex

    paths = {"train": args.train_csv, "val": args.val_csv, "test": args.test_csv}
    exists = {k: (v and os.path.exists(v)) for k, v in paths.items()}

    three_distinct = (exists["train"] and exists["val"] and exists["test"] and
                      len({paths["train"], paths["val"], paths["test"]}) == 3)
    two_distinct = (exists["train"] and exists["test"] and paths["train"] != paths["test"]
                    and (not exists["val"] or paths["val"] in {None, "", paths["train"], paths["test"]}))

    if three_distinct:
        ds = load_dataset("csv", data_files={
            "train": paths["train"],
            "validation": paths["val"],
            "test": paths["test"],
        })
        ds = ds.map(pick_text).map(map_label)
        # (não precisa estratificar aqui, já veio com splits prontos)

    elif two_distinct:
        base = load_dataset("csv", data_files={
            "train": paths["train"],
            "test":  paths["test"],
        })
        base = base.map(pick_text).map(map_label)

        # >>> Encode labels como ClassLabel antes do split estratificado
        if not isinstance(base["train"].features["labels"], ClassLabel):
            base = base.class_encode_column("labels")

        tmp = base["train"].train_test_split(
            test_size=0.1111, stratify_by_column="labels", seed=args.seed
        )
        ds = DatasetDict(train=tmp["train"], validation=tmp["test"], test=base["test"])

    else:
        # 1 CSV -> split 80/10/10
        full = load_dataset("csv", data_files={"train": paths["train"]})["train"]
        full = full.map(pick_text).map(map_label)

        # >>> Encode labels como ClassLabel antes de qualquer split estratificado
        if not isinstance(full.features["labels"], ClassLabel):
            full = full.class_encode_column("labels")

        tmp = full.train_test_split(
            test_size=0.2, stratify_by_column="labels", seed=args.seed
        )
        val_test = tmp["test"].train_test_split(
            test_size=0.5, stratify_by_column="labels", seed=args.seed
        )
        ds = DatasetDict(train=tmp["train"], validation=val_test["train"], test=val_test["test"])

    # Tokenização
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_len)

    drop_cols = [c for c in ds["train"].column_names if c not in ["text", "labels"]]
    ds_tok = ds.map(tok, batched=True, remove_columns=drop_cols).with_format("torch")

    return {"raw": ds, "tok": ds_tok, "tokenizer": tokenizer}




# ============================
# 4) Métricas
# ============================
ACC = evaluate.load("accuracy")
F1  = evaluate.load("f1")
PR  = evaluate.load("precision")
RC  = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, y = eval_pred
    yhat = np.argmax(logits, axis=1)
    out = {
        "accuracy": ACC.compute(predictions=yhat, references=y)["accuracy"],
        "macro_f1": F1.compute(predictions=yhat, references=y, average="macro")["f1"],
        "macro_precision": PR.compute(predictions=yhat, references=y, average="macro")["precision"],
        "macro_recall": RC.compute(predictions=yhat, references=y, average="macro")["recall"],
    }
    return out


# ============================
# 5) Class Weights (opcional)
# ============================
class WeightedTrainer(Trainer):
    """
    Trainer com CrossEntropyLoss ponderada por classe (quando args.class_weights='auto').
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def make_class_weights(ds_train, device) -> Optional[torch.Tensor]:
    labels = np.array(ds_train["labels"])
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(float)
    # peso ~ inverso da frequência (soma normalizada opcional)
    inv = counts.sum() / (counts + 1e-9)
    w = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
    return w


# ============================
# 6) Construção do modelo
# ============================

def build_model(args, device_map="auto"):
    load_kwargs = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID
    }

    # Config com 3 classes
    config = AutoConfig.from_pretrained(
        args.checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Quantização (API nova) – opcional
    quant_cfg = None
    if getattr(args, "load_in_8bit", False):
        # OBS: em Windows nativo o bitsandbytes costuma falhar; WSL2/Linux é recomendado.
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    if args.lora:
        # Carrega o backbone + cabeça de 3 classes, ignorando o tamanho diferente na cabeça antiga (2→3)
        base = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint,
            config=config,
            quantization_config=quant_cfg,      # None se não quiser 8-bit
            device_map=device_map if quant_cfg is not None else None,
            ignore_mismatched_sizes=True        # <<< ESSENCIAL
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
        return model

    # Sem LoRA (fine-tuning “cheio”)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        config=config,
        quantization_config=quant_cfg,
        device_map=device_map if quant_cfg is not None else None,
        ignore_mismatched_sizes=True            # <<< ESSENCIAL
    )
    return model


# ============================
# 7) Main
# ============================
def main():
    args = get_args()

    # Seeds / TF32
    from transformers.trainer_utils import set_seed
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dados
    data = build_datasets(args)
    ds_raw, ds_tok, tokenizer = data["raw"], data["tok"], data["tokenizer"]

    # Modelo
    device_map = "auto" if (args.lora and args.load_in_8bit) else None
    model = build_model(args, device_map=device_map)

    if args.grad_checkpoint:
        # gradient checkpointing só no modo "completo" ou se PEFT suportar
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # Pesos de classe
    class_weights = None
    if args.class_weights == "auto":
        # device a usar para os pesos
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_weights = make_class_weights(ds_tok["train"], device=device)

    # TrainingArguments
    scheduler = "cosine" if args.cosine else "linear"
    report_to = "none" if args.no_report_to else "auto"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.log_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=report_to,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=scheduler
    )

    # Trainer (ponderado se necessário)
    if class_weights is not None:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

    # Treino
    trainer.train()

    # Avaliação (val + test)
    print("\n== Validação ==")
    val_metrics = trainer.evaluate(ds_tok["validation"])
    for k, v in val_metrics.items():
        if k.startswith("eval_"):
            print(f"{k}: {v:.6f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    print("\n== Teste ==")
    test_metrics = trainer.evaluate(ds_tok["test"])
    for k, v in test_metrics.items():
        if k.startswith("eval_"):
            print(f"{k}: {v:.6f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    # (Opcional) relatório detalhado de classificação
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        preds = trainer.predict(ds_tok["test"])
        yhat = np.argmax(preds.predictions, axis=1)
        ytrue = preds.label_ids
        print("\n== Classification report (TEST) ==")
        print(classification_report(ytrue, yhat, target_names=[ID2LABEL[i] for i in range(NUM_LABELS)]))
        print("\n== Confusion matrix (TEST) ==")
        print(confusion_matrix(ytrue, yhat))
    except Exception as e:
        print(f"[AVISO] sklearn não disponível para classification_report/confusion_matrix: {e}")

    # Salvar melhor modelo + tokenizer
    save_dir = os.path.join(args.output_dir, "best")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\nTreino concluído.")
    print(f"Melhor modelo salvo em: {save_dir}")
    print("Use para inferência, por exemplo:")
    print(f"""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

ckpt = r"{save_dir}"
tok  = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
mdl  = AutoModelForSequenceClassification.from_pretrained(ckpt)

clf = pipeline("text-classification", model=mdl, tokenizer=tok, truncation=True, padding=True, top_k=None)
print(clf("A inflação acumulada foi de 4,3% em 2024, segundo o IBGE."))
""")


if __name__ == "__main__":
    main()
