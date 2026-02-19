from datasets import load_dataset
import pandas as pd
import re

# Load the dataset
print("Loading dataset...")
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.DataFrame(ds["train"])

# ── Placeholder replacements ──────────────────────────────────────────
replacements = {
    "{{Customer Support Email}}":   "support@shopease.com",
    "{{Customer Support Phone Number}}": "+1-800-555-0199",
    "{{Live Chat Support}}":        "live chat on our website",
    "{{Website URL}}":              "www.shopease.com",
    "{{Order Number}}":             "#ORD-XXXX",
    "{{Invoice Number}}":           "#INV-XXXX",
    "{{Date}}":                     "the specified date",
    "{{Date Range}}":               "the specified date range",
    "{{Delivery City}}":            "your city",
    "{{Delivery Country}}":         "your country",
    "{{Shipping Cut-off Time}}":    "5:00 PM",
    "{{Store Location}}":           "your nearest store",
    "{{Salutation}}":               "Dear Customer",
    "{{Client First Name}}":        "there",
    "{{Client Last Name}}":         "",
    "{{Refund Amount}}":            "the refund amount",
    "{{Money Amount}}":             "the specified amount",
    "{{Profile}}":                  "your profile",
    "{{Profile Type}}":             "your profile type",
    "{{Settings}}":                 "your account settings",
    "{{Online Order Interaction}}": "your order",
    "{{Online Payment Interaction}}": "your payment",
    "{{Online Navigation Step}}":   "the relevant page",
    "{{Online Company Portal Info}}": "our company portal",
    "{{Upgrade Account}}":          "account upgrade",
    "{{Account Type}}":             "your account type",
    "{{Account Category}}":         "your account category",
    "{{Account Change}}":           "account change",
    "{{Program}}":                  "the program",
}

def clean_text(text):
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    # Remove any remaining {{...}} just in case
    text = re.sub(r"\{\{.*?\}\}", "", text)
    # Clean extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning to both instruction and response
df["instruction"] = df["instruction"].apply(clean_text)
df["response"]    = df["response"].apply(clean_text)

# ── Sample dataset ────────────────────────────────────────────────────
# 50 rows per intent for a balanced knowledge base (~1350 rows total)
df_sampled = (
    df.groupby("intent", group_keys=False)
      .apply(lambda x: x.sample(min(50, len(x)), random_state=42), include_groups=False)
      .reset_index(drop=True)
)

# ── Rename and keep relevant columns ─────────────────────────────────
df_sampled = df_sampled[["instruction", "response", "category", "intent"]]
df_sampled.columns = ["query_text", "response_text", "category", "intent"]

# Add a document_id
df_sampled.insert(0, "document_id", range(1, len(df_sampled) + 1))

# ── Save to CSV ───────────────────────────────────────────────────────
df_sampled.to_csv("data/knowledge_base.csv", index=False)

print(f"Saved {len(df_sampled)} rows to data/knowledge_base.csv")
print(f"\nIntent distribution:\n{df_sampled['intent'].value_counts()}")
print(f"\nSample row:\n{df_sampled.iloc[0]}")