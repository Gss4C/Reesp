import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import marimo as mo
    return mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""## Dataset House Pricing""")
    return


@app.cell
def _(pd):
    df1 = pd.read_csv("./data/HP-ADVT/train.csv", header=0)
    df1.info()
    return (df1,)


@app.cell
def _(df1):
    df1.describe()
    return


@app.cell
def _(df1):
    df1
    return


@app.cell
def _(df1, plt):
    df1.hist(bins=30, column='SalePrice') #non funziona ma non capisco perche
    plt.title("Prezzi vendita case in $")
    plt.xlabel("Prezzo")
    plt.ylabel("Conteggio")
    #plt.show()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""## Dataset Boston House""")
    return


@app.cell
def _(pd):
    df2 = pd.read_csv("./data/BostonHP/boston.csv", header=0)
    df2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Decisione delle variabili

    - Id
    - YearBuilt
    - Heating
    - GarageCars
    - SalePrice
    - Fireplaces
    - LandSlope
    - Street
    - LotArea
    - OverallCond
    """
    )
    return


@app.cell
def _():
    import altair as alt
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
