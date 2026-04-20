# Trading Watchlist

This is the human-readable reference for the executable catalog in [config/live-data.yaml](config/live-data.yaml). The catalog contains 145 configured instruments: 7 indexes, 9 futures, 11 FX pairs, 5 crypto pairs, and 113 stock-type contracts (99 equities/ADRs and 14 ETFs).

## Focused Coverage

### Always-on core markets

The active universe always includes these 31 macro, benchmark, and crypto markets, plus symbols required by enabled pattern lists:

- Indexes: DAX, DJI, NDQ, SOX, SPX, VIX
- Futures: CL, ES, ESTX50, GC, HSI, MES, NQ, ZC, ZS
- FX: AUDJPY, AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY
- Crypto: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT

### Intraday focus

`bollinger_reversion` evaluates only: SPY, QQQ, GLD, ES, NQ, COR, DD, JPM, and SNDK.

### Priority qualification candidates

Every one of the 99 configured equity/ADR candidates is considered on each refresh. `watchlist_candidate_limit: 0` removes the fixed prefix cutoff; the shared 300-candidate discovery and qualification budgets still apply. Symbols already required by enabled pattern lists are included in the core instead. Remaining candidates must pass weekly/monthly trend qualification before entering the active snapshot.

## Full Catalog

### Indexes

| Display symbol | Instrument | IBKR contract |
| --- | --- | --- |
| DAX | DAX Index | DAX / EUREX / EUR |
| DXY | US Dollar Index | DXY / ICEUS / USD |
| DJI | Dow Jones Industrial Average | INDU / CME / USD |
| NDQ | Nasdaq 100 Index | NDX / NASDAQ / USD |
| SOX | PHLX Semiconductor Sector Index | SOX / PHLX / USD |
| SPX | S&P 500 Index | SPX / CBOE / USD |
| VIX | Cboe Volatility Index | VIX / CBOE / USD |

### Futures

| Symbol | Instrument | Exchange | Currency |
| --- | --- | --- | --- |
| CL | Crude Oil Futures | NYMEX | USD |
| ES | E-mini S&P 500 Futures | CME | USD |
| ESTX50 | Euro Stoxx 50 Futures | EUREX | EUR |
| GC | Gold Futures | COMEX | USD |
| HSI | Hang Seng Index Futures | HKFE | HKD |
| MES | Micro E-mini S&P 500 Futures | CME | USD |
| NQ | Nasdaq 100 Futures | CME | USD |
| ZC | Corn Futures | CBOT | USD |
| ZS | Soybean Futures | CBOT | USD |

### Forex

AUDJPY, AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY.

### ETFs

| Symbol | Instrument |
| --- | --- |
| ACWI | iShares MSCI ACWI ETF |
| EEM | iShares MSCI Emerging Markets ETF |
| EFA | iShares MSCI EAFE ETF |
| EWJ | iShares MSCI Japan ETF |
| EWZ | iShares MSCI Brazil ETF |
| FXI | iShares China Large-Cap ETF |
| GLD | SPDR Gold Shares |
| INDA | iShares MSCI India ETF |
| IWM | iShares Russell 2000 ETF |
| QQQ | Invesco QQQ ETF |
| SPY | SPDR S&P 500 ETF |
| TLT | iShares 20+ Year Treasury Bond ETF |
| VGK | Vanguard FTSE Europe ETF |
| VTI | Vanguard Total Stock Market ETF |

### US Equities And ADRs

AAPL, ABBV, ACGL, ADBE, ADI, AMD, AMZN, ASML, AXON, BX, CAH, CASY, CBOE, CCEP, CHRW, CMS, COR, COST, CRM, DD, DE, DELL, DOW, ECL, EME, ETR, FAST, FOX, GOOG, GOOGL, HCA, HLT, HPQ, HST, HUBB, HWM, IFF, INTC, ISRG, JKHY, JPM, KDP, KMB, KO, KR, LEN, LH, LLY, LRCX, MCD, META, MKC, MRVL, MSFT, MSTR, MTB, NEM, NFLX, NI, NKE, NVDA, ON, ORLY, PEP, PFE, PGR, PM, SHOP, SNDK, T, TE, TER, TMUS, TSLA, TSMC, UBER, UMC, V, VRTX, WELL, ZBRA.

`TSMC` intentionally resolves to the NYSE ADR contract `TSM`; `UMC` resolves to the NYSE ADR of United Microelectronics.

### European Equities

AKE, AXA, BC, BGN, BN, BOL, CVC, DSFIR, DTE, EDEN, ENI, FRE, GFC, GLE, LDO, LOTB, SAF, TKO.

### Crypto

| Display symbol | Instrument | Provider symbol |
| --- | --- | --- |
| BNBUSDT | BNB | BNBUSDT |
| BTCUSDT | Bitcoin | BTCUSDT |
| ETHUSDT | Ethereum | ETHUSDT |
| SOLUSDT | Solana | SOLUSDT |
| XRPUSDT | XRP | XRPUSDT |

## Pending Contract Mapping

`ALBPS` and `OESX` were present in the original text watchlist but are not configured. Their ticker strings alone do not identify a single IBKR contract. Add them after recording the intended issuer, exchange, and currency; this avoids failed historical-data requests for an ambiguous contract.

`DXY` remains in the catalog but is excluded from the always-on core: the paper Gateway returned no matching USD index contract. Re-enable it only after verifying the intended index mapping. DJI (`INDU` / CME) and SOX (`SOX` / PHLX) were verified through read-only contract-details requests.
