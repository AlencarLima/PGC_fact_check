from sklearn.metrics import confusion_matrix
import pandas as pd

labels = ["factual", "non-factual"]

# Matriz de confusão (contagens absolutas)
cm = confusion_matrix(golds, preds, labels=labels)
cm_df = pd.DataFrame(cm,
                     index=[f"gold_{l}" for l in labels],
                     columns=[f"pred_{l}" for l in labels])
print("Matriz de confusão (contagens):")
print(cm_df)

# Matriz normalizada por linha (≈ recall por classe)
cm_norm = confusion_matrix(golds, preds, labels=labels, normalize="true")
cm_norm_df = pd.DataFrame(cm_norm.round(3),
                          index=[f"gold_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels])
print("\nMatriz de confusão normalizada (por linha):")
print(cm_norm_df)

# (opcional) salvar em CSV
cm_df.to_csv("confusion_matrix_counts.csv")
cm_norm_df.to_csv("confusion_matrix_normalized.csv")
