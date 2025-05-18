<h1 align="center">E-Library-Agent</h1>

<h2 align="center">A virtual agent for your virtual books📚</h2>

<div align="center">
    <h3>If you find Code-RAGent useful, please consider to donate and support the project:</h3>
    <a href="https://github.com/sponsors/AstraBert"><img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA" alt="GitHub Sponsors Badge"></a>
</div>
<br>
<br>

E-library agent is a showcase project for [**ingest-anything**](https://github.com/AstraBert/ingest-anything), a versatile library to perform data ingestion into a vector database.

The main tasks that the e-library-agent can perform are:

- Build your library, progressively ingesting all the files you feed to it
- Retrieve information from the library
- Search for new books or papers on the internet

Beyond ingest-anything, the application is powered by [**LlamaIndex**](https://llamaindex.ai), [**Qdrant**](https://qdrant.tech) and [**Linkup**](https://linkup.so).

## Install and launch🚀

> _Required: [Docker](https://docs.docker.com/desktop/) and [docker compose](https://docs.docker.com/compose/)_

The first step, common to both the Docker and the source code setup approaches, is to clone the repository and access it:

```bash
git clone https://github.com/AstraBert/e-library-agent.git
cd e-library-agent/
```

Once there, you can follow this approach

- Add the `openai_api_key` and `linkup_api_key` variable in the [`.env.example`](./.env.example) file and modify the name of the file to `.env`. Get these keys:
  - [On OpenAI Platform](https://platform.openai.com/api-keys)
  - [On Linkup Dashboard](https://app.linkup.so/api-keys)

```bash
mv .env.example .env
```

- You can now launch the containers with the following commands:

```bash
docker compose up qdrant -d
docker compose up app -d
```

You will see the application running on http://localhost:8000. Depending on your connection and on your hardware, the set up might take some time (up to 15 mins to set up) - but this is only for the first time your run it!

## How it works

### Database services

- **Qdrant** is used to store vectors coming from the ingestion of your books. It is updated every time you ingest one or more files!

### Workflow

**Ingestion:**

- The text is extracted from the uploaded files using the **ingest-anything** extraction framework.
- Chunking is also included in this framework, and uses [NeuralChunker](https://docs.chonkie.ai/python-sdk/chunkers/neural-chunker) by [Chonkie](https://chonkie.ai).
- The chunked files are embedded by OpenAI **text-embedding-3-small** into 1536-dimensional dense embeddings (still within the ingest-anything framework)
- The embeddings are loaded into a Qdrant collection (last step of ingest-anything ingestion)

**Retrieval of information from library:**

- The IngestAgent that serves as librarian, based on a ReAct agent, retrieves the context from the vector database using a RAG tool
- The context is evaluated
- If the context is sufficiently relevant, it is used as a base to the response returned to the user

**Web search:**

- The web searching assistant, based on a function-calling agent, routes your query to a deepsearching tool backed by Linkup
- The response from the web search is evaluated
- If the context is sufficiently relevant, it is used as a base to the response returned to the user

## Contributing

Contributions are always welcome! Follow the contributions guidelines reported [here](CONTRIBUTING.md).

## License and rights of usage

The software is provided under MIT [license](./LICENSE).
