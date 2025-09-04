import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    df_i = pd.read_csv("/home/jonathan/Documenti/repo/Reesp/data/HP-ADVT/train.csv")
    df_i
    return (df_i,)


@app.cell
def _(df_i):
    df_f = df_i[["Id", "YearBuilt", "Heating", "GarageCars", "SalePrice", "Fireplaces", "LandSlope", "Street", "LotArea", "OverallCond"]]
    df_f[["Id", "YearBuilt", "Heating", "GarageCars", "SalePrice", "Fireplaces", "LandSlope", "Street", "LotArea", "OverallCond"]]
    return (df_f,)


@app.cell
def _(df_i):
    df_i.describe()
    return


@app.cell
def _(df_f):
    df_f.describe()
    return


@app.cell
def _(df_f):
    print(df_f.get("YearBuilt")[15])
    return


@app.cell
def _(df_f):
    ciao = df_f.get(['YearBuilt', 'OverallCond', 'Street'])[50:500].to_dict()
    ciao
    return (ciao,)


@app.cell
def _(ciao):
    ciao["YearBuilt"][55]
    return


@app.cell
def _():
    cacca = ['cacca', 'cacchina', 'caccone']
    oput = ', '.join(cacca)
    return (oput,)


@app.cell
def _(oput):
    print(oput.strip())
    return


@app.cell
def _(df_f):
    for idx, row in df_f.iterrows():
        print(idx +1 , row['Id'])
    
    return


@app.cell
def _():
    return


@app.cell
def _():
    prova_diz = {
        'classe': 5,
        'classona': 11
    }
    prova_diz.get('classe')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
