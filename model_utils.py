import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-multilingual-cased"
LABELS     = ["Negative", "Neutral", "Positive"]
MAX_LEN    = 128

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def get_model(num_labels=3):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

def predict(text, model, tokenizer, device):
    model.eval()
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
    probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_label = LABELS[np.argmax(probs)]
    confidence = float(np.max(probs))
    all_scores = {LABELS[i]: float(probs[i]) for i in range(3)}
    return pred_label, confidence, all_scores

def get_token_importance(text, model, tokenizer, device):
    model.eval()
    enc = tokenizer(text, max_length=MAX_LEN, truncation=True, return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    embeddings = model.bert.embeddings(input_ids)
    embeddings.retain_grad()
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    pred_class = outputs.logits.argmax(dim=1).item()
    outputs.logits[0, pred_class].backward()
    saliency = embeddings.grad[0].norm(dim=-1).cpu().detach().numpy()
    tokens   = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    result   = []
    for tok, score in zip(tokens, saliency):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        display = tok.replace("##", "")
        if display.strip():
            result.append((display, float(score)))
    if result:
        scores = np.array([s for _, s in result])
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        result = [(t, float(s)) for (t, _), s in zip(result, scores)]
    return result
