import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from transformers import BertModel
    from transformers import BertTokenizer, BertForSequenceClassification
    return BertForSequenceClassification, BertModel, BertTokenizer, torch


@app.cell
def _(BertTokenizer):
    #processo di tokenizzazione della frase e del contesto, necessario per il funzionamento di BERT

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    textone = "BERT works better in english than in any other language"
    tokens = tokenizer.tokenize(textone)

    print(tokens)
    return (tokenizer,)


@app.cell
def _(BertForSequenceClassification, tokenizer, torch):
    # testo import di modello

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    text = "This movie was amazing!"
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    print(predictions)
    return


@app.cell
def _(BertModel, tokenizer):
    # Visualizing Attention Weights

    modello = BertModel.from_pretrained('bert-base-uncased')

    testo = "BERT's attention mechanism is fascinating."
    inputti = tokenizer(testo, return_tensors='pt', padding=True, truncation=True)
    outputti = modello(**inputti, output_attentions=True)

    attention_weights = outputti.attentions
    print(attention_weights)
    return (modello,)


@app.cell
def _(modello):
    from torchsummary import summary
    summary(modello)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
