"""Score distribution -> prices (Phase 2). Nothing here yet.

Planned modules:
  markets.py  contract specs that mirror Kalshi's rules exactly:
                KXNFLGAME   YES pays if the team wins; a tie settles 50/50,
                            so fair = P(win) + 0.5 * P(tie)
                KXNFLSPREAD "<team> wins by over N.5" -> P(margin > N.5)
                KXNFLTOTAL  "over N.5 points"          -> P(total > N.5)
  fair.py     joint final-score samples -> fair price for every contract.
              One simulation run prices every rung, so prices are consistent
              with each other by construction.
  quote.py    fair price -> bid/ask (margin, 1c ticks) and suspension while a
              play is live. Kalshi itself never suspends; our quotes do, for
              realism. The benchmark always uses the fair price.
"""
