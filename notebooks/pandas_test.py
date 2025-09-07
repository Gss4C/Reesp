import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    dictest = {
        '1': [13,54,3],
        '2': [2,2]
    }
    return


@app.cell
def _():
    return


@app.cell
def _():
    altro_test = [
        {
        'Id': 45,
        'cacca': 42},
        {
        'Id': 95,
        'cacca': 22
        }]
    return (altro_test,)


@app.cell
def _(altro_test, pd):
    pd.DataFrame(altro_test)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    img_dict =    { "1445": [
            "data/images/house_data/din_1445.jpg"
        ],
        "1511": [
            "data/images/house_data/din_1511.jpg"
        ],
        "1412": [
            "data/images/house_data/bed_1412.jpg"
        ],
        "1558": [
            "data/images/house_data/din_1558.jpg"
        ],
        "1463": [
            "data/images/house_data/din_1463.jpg"
        ],
        "1489": [
            "data/images/house_data/din_1489.jpg"
        ],
        "897": [
            "data/images/house_data/bed_897.jpg"
        ],
        "1561": [
            "data/images/house_data/din_1561.jpg"
        ],
        "1495": [
            "data/images/house_data/din_1495.jpg"
        ],
        "71": [
            "data/images/house_data/living_71.jpg",
            "data/images/house_data/bed_71.jpg"
        ],
        "1449": [
            "data/images/house_data/din_1449.jpg"
        ],
        "1499": [
            "data/images/house_data/din_1499.jpg"
        ],
        "64": [
            "data/images/house_data/din_64.jpg",
            "data/images/house_data/kitchen_64.jpg"
        ],
        "1579": [
            "data/images/house_data/din_1579.jpg"
        ],
        "1569": [
            "data/images/house_data/din_1569.jpg"
        ],
        "1551": [
            "data/images/house_data/din_1551.jpg"
        ],
        "1559": [
            "data/images/house_data/din_1559.jpg"
        ]
    }
    return (img_dict,)


@app.cell
def _(img_dict):
    final_list = []
    for key, value in img_dict.items():
        for single_path in value:
            #print(single_path)
            final_list.append({
                'iD': key,
                'image_path': single_path
            })
    print(final_list)
    return (final_list,)


@app.cell
def _(final_list, pd):
    pd.DataFrame(final_list)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
