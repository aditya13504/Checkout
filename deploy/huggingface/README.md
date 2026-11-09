---
title: Checkout
emoji: 🛒
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: An RL value shopper buys from 1.18M real products, live
---

# Checkout

Watch a value shopper, trained with reinforcement learning on the WebShop benchmark, shop a store of 1,181,430 real products, one step at a time. Pick one of the 500 test requests, which the store scores out of 100, or type your own request. The store itself is open too: shop the same request yourself and compare.

The code, the training pipeline and the report are at [github.com/aditya13504/Checkout](https://github.com/aditya13504/Checkout).

After a period without visitors the Space sleeps. The first visit after that wakes it, which takes a minute or two while the store loads.
