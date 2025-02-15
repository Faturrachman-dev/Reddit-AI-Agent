The `text-embedding-004` model generates state-of-the-art embeddings for words, phrases, and sentences. The resulting embeddings can then be used for tasks such as semantic search, text classification, and clustering, among many others. For more information on embeddings, read our [research paper](https://deepmind.google/research/publications/85521/).

## What are embeddings?

Embeddings capture semantic meaning and context, which results in text with similar meanings having "closer" embeddings. For example, the sentence "I took my dog to the vet" and "I took my cat to the vet" would have embeddings that are close to each other in the vector space since they both describe a similar context.

You can use embeddings to compare different texts and understand how they relate. For example, if the embeddings of the text "cat" and "dog" are close together you can infer that these words are similar in meaning, context, or both. This enables a variety of [common AI use cases](https://ai.google.dev/gemini-api/docs/embeddings#use-cases).

## Generate embeddings

Use the [`embedContent` method](https://ai.google.dev/api/embeddings#method:-models.embedcontent) to generate text embeddings:

[Python](https://ai.google.dev/gemini-api/docs/embeddings#python)[Node.js](https://ai.google.dev/gemini-api/docs/embeddings#node.js)[curl](https://ai.google.dev/gemini-api/docs/embeddings#curl)[Go](https://ai.google.dev/gemini-api/docs/embeddings#go)

```
<span>from</span><span> </span><span>google</span><span> </span><span>import</span> <span>genai</span>
<span>from</span><span> </span><span>google.genai</span><span> </span><span>import</span> <span>types</span>

<span>client</span> <span>=</span> <span>genai</span><span>.</span><span>Client</span><span>(</span><span>api_key</span><span>=</span><span>"<devsite-var rendered="" translate="no" is-upgraded="" scope="GEMINI_API_KEY" tabindex="0"><span><var spellcheck="false" is-upgraded="" data-title="Edit GEMINI_API_KEY" aria-label="Edit GEMINI_API_KEY">GEMINI_API_KEY</var></span></devsite-var>"</span><span>)</span>

<span>result</span> <span>=</span> <span>client</span><span>.</span><span>models</span><span>.</span><span>embed_content</span><span>(</span>
        <span>model</span><span>=</span><span>"text-embedding-004"</span><span>,</span>
        <span>contents</span><span>=</span><span>"What is the meaning of life?"</span><span>)</span>

<span>print</span><span>(</span><span>result</span><span>.</span><span>embeddings</span><span>)</span>
```

### Use cases

Text embeddings are used in a variety of common AI use cases, such as:

-   **Information retrieval:** You can use embeddings to retrieve semantically similar text given a piece of input text.
    
    [Document search tutorial](https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/document_search.ipynb)
    
-   **Clustering:** Comparing groups of embeddings can help identify hidden trends.
    
    [Embedding clustering tutorial](https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/clustering_with_embeddings.ipynb)
    
-   **Vector database:** As you take different embedding use cases to production, it is common to store embeddings in a vector database.
    
    [Vector database tutorial](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings)
    
-   **Classification:** You can train a model using embeddings to classify documents into categories.
    
    [Classification tutorial](https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/text_classifier_embeddings.ipynb)
    

### Gemini embeddings models

The Gemini API offers two models that generate text embeddings:

-   [Text Embeddings](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding)
-   [Embeddings](https://ai.google.dev/gemini-api/docs/models/gemini#embedding)

Text Embeddings is an updated version of the Embedding model that offers elastic embedding sizes under 768 dimensions. Elastic embeddings generate smaller output dimensions and potentially save computing and storage costs with minor performance loss.

Use Text Embeddings for new projects or applications.