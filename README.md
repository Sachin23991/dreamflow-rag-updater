# dreamflow-rag-updater

Automated RAG updater that continuously ingests fresh Stack Overflow Q&A, builds sharded RAG components, and uploads them to Hugging Face for use in DreamFlow or any RAG-based LLM system.

## Overview

dreamflow-rag-updater is a small orchestrator around an existing RAG stack that turns Stack Overflow into a live, ever-growing retrieval corpus. It periodically scrapes new questions and answers, appends them to a train buffer, builds RAG artifacts, merges them into the existing index, and pushes everything to Hugging Face in a fully automated loop.

## Features

- **Automated Stack Overflow Scraping** - Fetches fresh Q&A data via `fetch_stackoverflow_qa()`
- **Append-Only Train Buffer** - Incremental `train.jsonl` managed by `append_to_train_buffer()` and `clear_train_buffer()`
- **RAG Component Building** - Converts raw Q&A into embedding-ready artifacts with `build_rag_component()`
- **Sharded RAG Merging** - Merges and uploads versioned RAG components to Hugging Face via `merge_and_upload_rag()`
- **Automated Pipeline Orchestration** - Single `run()` entrypoint that wires the entire pipeline for local or scheduled execution
- **Stateless Between Runs** - Buffer cleanup ensures each run only processes new data

## Architecture

The updater is a thin wrapper around several focused, reusable modules:

### Scraping
- **`stackoverflow_scraper.py`**: Implements `fetch_stackoverflow_qa()` to pull structured Q&A data

### Buffering
- **`train_buffer_manager.py`**
  - `append_to_train_buffer(qa)`: Appends new samples into `train.jsonl`
  - `clear_train_buffer()`: Clears the buffer after a successful run

### RAG Building
- **`rag_builder.py` / `rag_component_builder.py`**
  - `build_rag_component()`: Builds a temporary RAG component from the train buffer

### Merging & Upload
- **`hf_rag_merger.py` / `hf_rag_uploader.py`**
  - `merge_and_upload_rag(temp_rag)`: Merges the new component into existing RAG and uploads to Hugging Face

### Search & Inference
- **`vector_db.py`**, **`inference_search.py`**: Vector store logic and search utilities
- **`datasets/`**, **`rag/`**: Materialized datasets and RAG artifacts used by DreamFlow

### Orchestration
- **`main.py`**: Implements `run()` function that orchestrates the entire pipeline

## RAG Update Flow

The default pipeline run follows this sequence:

```
1. Scrape     -> fetch_stackoverflow_qa()
   |
2. Buffer     -> append_to_train_buffer(qa)
   |
3. Build RAG  -> build_rag_component()
   |
4. Merge & Upload -> merge_and_upload_rag(temp_rag)
   |
5. Cleanup    -> clear_train_buffer()
```

This entire flow is encapsulated in `run()` and can be triggered as often as needed (hourly, daily, etc.).

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Hugging Face account with repository/dataset access
- Stack Overflow API access (if required by your scraper)
- Git

### Clone & Install

```bash
git clone https://github.com/Sachin23991/dreamflow-rag-updater.git
cd dreamflow-rag-updater
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file or configure environment variables:

```bash
# Hugging Face
HF_TOKEN=your_huggingface_token_here
HF_DATASET_ID=your_dataset_id
HF_REPO_ID=your_repo_id

# Stack Overflow (if required)
SO_API_KEY=your_stackoverflow_key

# Optional: Batch size, logging level
BATCH_SIZE=100
LOG_LEVEL=INFO
```

## Usage

### Local Execution

Run the full pipeline once:

```bash
python main.py
```

This executes:
1. Fresh Stack Overflow scrape
2. Data appended to `train.jsonl`
3. RAG component build from buffer
4. Merge with existing RAG on Hugging Face
5. Buffer cleanup

### Scheduled Execution (GitHub Actions)

The repository includes a pre-configured GitHub Actions workflow for automated scheduled runs.

#### Setup

1. **Add Repository Secrets**
   - Go to Settings > Secrets and variables > Actions
   - Add secrets:
     - `HF_TOKEN`: Your Hugging Face write access token
     - `SO_API_KEY`: Stack Overflow API key (if needed)

2. **Configure Workflow Schedule**
   - Edit `.github/workflows/rag-update.yml` to adjust frequency
   - Common cron patterns:
     - `0 * * * *` - Every hour
     - `0 0 * * *` - Daily at midnight  
     - `0 0 * * 0` - Weekly on Sunday
     - `*/30 * * * *` - Every 30 minutes

#### Workflow File: `.github/workflows/rag-update.yml`

```yaml
name: Hourly RAG Update

on:
  schedule:
    - cron: '0 * * * *'  # Run every hour
  workflow_dispatch:      # Allow manual trigger

jobs:
  update-rag:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run RAG update pipeline
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          SO_API_KEY: ${{ secrets.SO_API_KEY }}
        run: python main.py
      
      - name: Commit changes (optional)
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add -A
          git commit -m "RAG update: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" || true
          git push
```

## Monitoring & Debugging

### View Workflow Runs

1. Go to your repository
2. Click **Actions** tab
3. Select **Hourly RAG Update** workflow
4. Click any run to see detailed logs

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `HF_TOKEN not set` | Missing Hugging Face token | Add `HF_TOKEN` to repository secrets |
| `Stack Overflow API rate limit` | Too many requests | Increase cron interval or reduce batch size |
| `Connection timeout` | Network issues | Check internet, retry manually |
| `Merge conflict in RAG` | Multiple concurrent updates | Ensure only one workflow runs at a time |
| `Out of memory` | Large batch size | Reduce `BATCH_SIZE` in `.env` |

### Enable Debug Logging

Modify `.env`:

```bash
LOG_LEVEL=DEBUG
```

Or run locally with verbose output:

```bash
python main.py --verbose
```

## Use Cases

- **Coding Assistants**: Keep AI-powered developer assistants synchronized with latest Stack Overflow solutions
- **DreamFlow Backends**: Automated RAG for production DreamFlow deployments
- **Custom LLM Systems**: Any system needing continuously-refreshed programming knowledge
- **Documentation Bots**: Auto-update knowledge bases for support systems
- **Code Search**: Maintain always-current searchable corpus of programming Q&A

## Performance Considerations

- **Update Frequency**: Hourly updates balance freshness vs. API costs
- **Batch Size**: Tune `BATCH_SIZE` in `.env` to control memory usage
- **Sharding**: RAG components are automatically sharded for efficient retrieval
- **Deduplication**: Duplicate Q&A is automatically filtered
- **Incremental Growth**: Only new data is processed each run

## Troubleshooting

### Pipeline Fails at Scraping Step

```bash
# Test scraper directly
python -c "from stackoverflow_scraper import fetch_stackoverflow_qa; print(fetch_stackoverflow_qa())"
```

### RAG Build Fails

```bash
# Check train.jsonl exists
ls -lah train.jsonl

# Inspect first few samples
head -5 train.jsonl
```

### Hugging Face Upload Issues

```bash
# Test upload manually
python -c "from hf_rag_uploader import test_upload; test_upload()"
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and test locally
4. Commit with clear messages (`git commit -m "Add feature: description"`)
5. Push to your fork and open a Pull Request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or feature requests, please open an issue on [GitHub Issues](https://github.com/Sachin23991/dreamflow-rag-updater/issues).

## Related Projects

- [DreamFlow](https://github.com/Sachin23991/dreamflow) - Main AI platform using this RAG updater
- [Hugging Face Datasets](https://huggingface.co/datasets) - Dataset hosting platform
- [Stack Overflow API](https://stackexchange.com/apis) - Data source

---

**Last Updated**: December 2024  
**Maintainer**: [Sachin23991](https://github.com/Sachin23991)
