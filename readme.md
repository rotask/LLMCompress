# LLMCompress Summary Feature

This branch introduces a new feature that uses a summary as side information for improved compression.

## Usage

When running the script, you can now provide a summary using the `--summary` argument:

```
python main.py --summary "Your summary text here"
```

If not provided, it defaults to "This is a summary".

## How it works

The summary is used as additional context during both compression and decompression, potentially leading to better compression ratios by providing relevant information about the content being compressed.

## Note

Experimental feature - not fully functional.
