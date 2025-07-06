An explanation on how we add models to Bumblebee (copied from https://x.com/sean_moriarity/status/1715758666001928613)

The first thing to note is that almost all of the models have significant overlap in implementation details. A transformer is a transformer. There are different "families" of transformers, and slight implementation differences, but the high-level details are the same. That means it's really easy to abstract out most of the implementation details into generic functions. Jonatan did a really good job of pulling the generic details out into helper functions for us in Bumblebee, so implementing new models is just a matter of determining which variations a particular model uses. e.g. Does the model use rotary embeddings? Do they used Grouped Query Attention? Is it encoder-decoder or decoder-only?

The actual process of adding a model is essentially 4 steps:

1. First we create a mapping for the model here: https://github.com/kf8a/bumblebee/blob/main/lib/bumblebee.ex#L82

This is where we tell Bumblebee how to map model names in HuggingFace's config.json to an actual Bumblebee module. Each model family has it's own module, and within that module we implement different model function heads for the different types of models. You can see this if you look at Llama for example: https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/text/llama.ex#L162

We match on the model architecture (e.g. base, causal language modeling, and sequence classification) and then for each architecture we have an implementation. You'll notice here that each implementation shares a `core` implementation which is the meat of the model. If you look at the HuggingFace implementation, `core` is identical to their base model implementation, and then each architectural variation is just a different head on top of the base model.

2. The next step is to map the configuration to the HuggingFace configuration. We slightly alter their configuration names for consistency, but it's as easy as looking at the model config (e.g. LlamaConfig), defining the options in the module like so: https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/text/llama.ex#L4

Defining a struct for the module with those options (each model is a struct with model configuration): https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/text/llama.ex#L115

And then implementing the configuration conversion: https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/text/llama.ex#L357

The conversion is necessary because we have slight variations in config names so this is a normalization step. The conversion defines a mapping from HF config name to Bumblebee config option.

3. The next step is to actually implement the Axon version of the model given the model "spec" or configuration. For most models, you can base your implementation off of another model implementation already in the repository and just change some minor details. The implementations map more or less directly to the PyTorch implementation in the HuggingFace repo. Each PyTorch module (class) is analogous to a function in Bumblebee.

If we look at Llama, you'll see the base model or "core" has an embedder, a decoder, and then applies normalization on the output: https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/text/llama.ex#L238

This is a common pattern with decoder-only transformers. The bulk of the model is the "decoder", which actually just consists of some number of transformer "blocks" as we call them, given by the model configuration: https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/text/llama.ex#L299

This "blocks" function just implements the repeated application of transformer layers or blocks defined here: https://github.com/elixir-nx/bumblebee/blob/main/lib/bumblebee/layers/transformer.ex#L292

Which itself applies the actual multi-head attention, the feed-forward network, normalization, etc. 

Most of your work for a new model will actually be just determining the right configuration for the transformer blocks. You can control things like what type of normalization to apply (e.g. regular layer normalization or RMS norm), what the FFN should look like in the block, should normalization be applied before or after each group of layers, whether or not to use rotary embedding, etc. The transformer blocks implement most of the variants you'll find in transformer models, so you shouldn't ever really have to mess with the actual block implementations.

4. The final step is to define the parameter mapping. This is how we map PyTorch parameters to Axon parameters. It's pretty simple, you just take the Axon parameter names, and define how they should map to PyTorch parameter names in the PyTorch model. During conversion, Bumblebee will use this mapping to convert the parameter map into something the Axon model can use.

As for testing implementations, we typically look at the HuggingFace examples for a model and check our implementation outputs match those example outputs. If they don't, I usually debug intermediate outputs using Axon hooks.

That's pretty much it! I'm happy to help anybody add new models to Bumblebee. We're missing a lot from the HF repo - a lot of which should be trivial to add - so we could use the help!
