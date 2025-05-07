# Clone the repos
```bash
git clone git@github.com:METR/eval-analysis-public.git
git clone https://github.com/METR/hcast-public
git clone git@github.com:METR/RE-Bench.git
git clone git@github.com:poking-agents/modular-public.git
git clone git@github.com:METR/vivaria.git
```

# Install uv and sync
```bash
curl -fsSL https://get.uv.dev | bash
uv sync
```

# Download the dataset
```bash
wget "https://zenodo.org/records/8136017/files/data.zip?download=1" -O cybersecurity_dataset_v4.zip
unzip cybersecurity_dataset_v4.zip
```

```
brew install sphinx-doc
```