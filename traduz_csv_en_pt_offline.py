import argparse, re
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# ---- Argos Translate (offline) ----
import argostranslate.package, argostranslate.translate

CAND_TEXT_COLS = ["sentence","text","utterance","claim","statement","Sentence","Text","Utterance","Claim","Statement"]

def guess_text_column(df: pd.DataFrame, user_col: str = None) -> str:
    if user_col:
        low = {c.lower(): c for c in df.columns}
        return low.get(user_col.lower(), user_col) if user_col in df.columns or user_col.lower() in low else (_ for _ in ()).throw(KeyError(f"Coluna '{user_col}' não encontrada: {list(df.columns)}"))
    low = {c.lower(): c for c in df.columns}
    for cand in CAND_TEXT_COLS:
        if cand in df.columns: return cand
        if cand.lower() in low: return low[cand.lower()]
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]): return c
    raise KeyError(f"Não foi possível inferir a coluna de texto. Colunas: {list(df.columns)}")

def split_into_chunks(text: str, max_chars: int = 1200) -> List[str]:
    if not isinstance(text, str): text = str(text) if text is not None else ""
    text = text.strip()
    if len(text) <= max_chars: return [text]
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf: chunks.append(buf)
            buf = s
    if buf: chunks.append(buf)
    final=[]
    for c in chunks:
        if len(c) <= max_chars: final.append(c)
        else:
            for i in range(0,len(c),max_chars): final.append(c[i:i+max_chars])
    return final

def ensure_en_pt_installed():
    # baixa/instala o pacote en->pt se necessário
    available = argostranslate.package.get_available_packages()
    have = argostranslate.translate.get_installed_languages()
    if any(lang.code=="pt" for lang in have) and any(lang.code=="en" for lang in have):
        return
    pkg = next((p for p in available if p.from_code=="en" and p.to_code=="pt"), None)
    if pkg is None:
        # atualiza índice e tenta de novo
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next((p for p in available if p.from_code=="en" and p.to_code=="pt"), None)
    if pkg is None:
        raise RuntimeError("Pacote Argos en→pt não encontrado.")
    argostranslate.package.install_from_path(pkg.download())

def translate_texts(texts: List[str]) -> List[str]:
    ensure_en_pt_installed()
    installed = argostranslate.translate.get_installed_languages()
    en = next(l for l in installed if l.code=="en")
    pt = next(l for l in installed if l.code=="pt")
    translator = en.get_translation(pt)
    out=[]
    for t in tqdm(texts, desc="Traduzindo (offline)", unit="linhas"):
        if not t: out.append(""); continue
        chunks = split_into_chunks(t)
        trans_chunks = [translator.translate(c) for c in chunks]
        out.append(" ".join(trans_chunks).strip())
    return out

def process_file(input_path: Path, out_suffix: str, text_col: str = None):
    print(f"\n>> Lendo: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="latin-1")
    tgt = guess_text_column(df, text_col)
    print(f"Coluna de texto: {tgt}")
    texts = df[tgt].fillna("").astype(str).tolist()
    df["text_pt"] = translate_texts(texts)
    out = input_path.with_suffix("")
    out = Path(f"{out}{out_suffix}.csv")
    df.to_csv(out, index=False)
    print(f"OK: salvo em {out}")

def main():
    ap = argparse.ArgumentParser(description="Traduz CSV(s) EN->PT (offline, Argos) adicionando coluna text_pt")
    ap.add_argument("inputs", nargs="+", help="Arquivos .csv a traduzir")
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--suffix", default=".pt")
    args = ap.parse_args()
    for p in args.inputs:
        pth = Path(p)
        if not pth.exists():
            print(f"[AVISO] Arquivo não encontrado: {pth}")
            continue
        process_file(pth, out_suffix=args.suffix, text_col=args.text_col)

if __name__ == "__main__":
    main()
