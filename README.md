# uk gilt screener

a streamlit app for finding tax-efficient uk gilts. it joins the dmo 'gilts in issue' list with daily prices from dividenddata.co.uk to calculate post-tax yields for higher rate taxpayers.

capital gains on gilts are tax-free. this tool helps find low-coupon gilts trading below par to maximize that tax-free return.

## quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

the app will automatically scrape the latest prices on startup.

## automated updates

there's a github action that runs daily at 6pm to scrape the latest prices and update the `data/processed/gilts_latest.parquet` file.

## development

run tests:

```bash
pytest
```
