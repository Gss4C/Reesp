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
def _():
    return


if __name__ == "__main__":
    app.run()
