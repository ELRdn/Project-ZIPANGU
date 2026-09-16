# C-I training lane

`C-I` remains the configured 9B follow-up lane. It is not deleted by the K-I migration. New C-I runs must use a model-specific config that points at `ZIPANGU-C-I-9B` and an I-generation shared recipe, and must keep `safety.training_allowed: false` until an explicit human GO.
