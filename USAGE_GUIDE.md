# Gmail LLM Labeler - Usage Guide

## Overview

This guide explains how to use the gmail-llm-labeler with custom configurations for automatic email categorization using local Ollama LLM models.

## Configuration Files

### Testing Configurations (Dry-Run Mode)

**`config_test_3b.yaml`** - Test with Qwen 2.5 3B model
- 20 emails maximum
- Dry-run mode (no labels applied)
- Faster processing
- Good for initial testing

**`config_test_7b.yaml`** - Test with Qwen 2.5 7B model
- 20 emails maximum
- Dry-run mode (no labels applied)
- Slower but more accurate
- Good for comparison with 3B

### Production Configurations (Labels Applied)

**`config_production_3b.yaml`** - Production with Qwen 2.5 3B
- Up to 5000 emails
- Labels WILL BE APPLIED
- Faster processing
- Recommended for large volumes

**`config_production_7b.yaml`** - Production with Qwen 2.5 7B
- Up to 5000 emails
- Labels WILL BE APPLIED
- Higher accuracy
- Recommended when you have time

---

## Email Categories

### **recruiter**
Cold outreach from recruiters you've never worked with
- Persistent follow-ups
- Generic job opportunities
- Template messages

### **automarketing**
Promotional emails from companies you bought from
- Unwanted marketing
- Sales and special offers
- Automated campaigns

### **charity_activism**
Emails from nonprofits and causes you support
- Fundraising appeals
- Impact reports
- Advocacy campaigns
- NOT donation receipts (those go to "action")

### **newsletter**
Content you subscribe to and read regularly
- Free and paid newsletters
- Industry insights
- Trusted sources
- Includes company newsletters you read

### **action**
Automated transactional emails about YOUR actions
- Order confirmations
- Shipping notifications
- Receipts and payment confirmations
- Donation receipts from charities
- Password resets, verification codes
- NOT conversations with real people

### **main**
Everything else
- Personal emails from friends/family
- Work communications
- Recruiters you've worked with before
- Real human correspondence
- Edge cases

---

## Quick Start

### Step 1: Verify Ollama Models

Check that both models are downloaded:

```bash
ollama list
```

You should see:
- `qwen2.5:3b`
- `qwen2.5:7b`

If missing, download with:
```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

### Step 2: Test with 3B Model (Dry-Run)

```bash
cd /Users/gmarceau/code/gmail-llm-labeler
gmail-pipeline run --config config_test_3b.yaml
```

**What happens:**
- Processes 20 emails from your inbox
- Shows categorization results
- NO labels applied (dry-run)
- Creates metrics file

**Expected output:**
```
INFO - Extracting 20 emails from Gmail...
INFO - Categorized email as 'recruiter': "Cold outreach from technical recruiter"
INFO - Categorized email as 'newsletter': "Weekly tech digest from source you subscribe to"
...
INFO - Pipeline completed. 20 emails processed.
```

### Step 3: Review Results

Check metrics:
```bash
gmail-pipeline show-metrics
```

Or view the metrics file directly:
```bash
cat ~/.local/share/gmail-llm-labeler/pipeline_metrics_test_3b.json
```

Check logs for detailed categorization:
```bash
tail -20 ~/.local/share/gmail-llm-labeler/logs/llm_interactions.jsonl
```

### Step 4: Test with 7B Model (Dry-Run)

```bash
gmail-pipeline run --config config_test_7b.yaml
```

**Note:** This will process the SAME 20 emails as the 3B test (uses same database), allowing direct comparison.

### Step 5: Compare Models

View both metrics files:
```bash
cat ~/.local/share/gmail-llm-labeler/pipeline_metrics_test_3b.json
cat ~/.local/share/gmail-llm-labeler/pipeline_metrics_test_7b.json
```

**Compare:**
- Categorization accuracy (did they choose the right categories?)
- Processing speed (3B is faster)
- Explanation quality (7B may provide better reasoning)

### Step 6: Run Production (When Ready)

**Choose your model:**

**Option A: 3B Model (Faster)**
```bash
gmail-pipeline run --config config_production_3b.yaml
```

**Option B: 7B Model (More Accurate)**
```bash
gmail-pipeline run --config config_production_7b.yaml
```

⚠️ **WARNING:** Production configs have `dry_run: false` - labels WILL BE APPLIED!

---

## Common Operations

### Validate Configuration

Before running, validate your config:
```bash
gmail-pipeline validate-config config_test_3b.yaml
```

### Generate Sample Config

To see default configuration:
```bash
gmail-pipeline generate-config --output sample_config.yaml
```

### View Metrics from Last Run

```bash
gmail-pipeline show-metrics
```

### Check Ollama Status

Ensure Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### Test Ollama Model

Quick test of a model:
```bash
ollama run qwen2.5:3b "Categorize this: 'Hi, I saw your profile and think you'd be a great fit for...'"
```

---

## Understanding the Output

### During Processing

```
INFO - Using Ollama at http://localhost:11434/v1 with model qwen2.5:3b
INFO - Extracting emails from Gmail with query: in:inbox
INFO - Processing batch 1/1 (20 emails)
INFO - Categorized email <ID> as 'recruiter': Cold recruiter outreach
INFO - Categorized email <ID> as 'action': Order confirmation
...
```

### Metrics File Structure

```json
{
  "total_emails": 20,
  "categories": {
    "recruiter": 3,
    "automarketing": 5,
    "newsletter": 4,
    "action": 6,
    "main": 2
  },
  "processing_time": "45.2s",
  "errors": 0
}
```

### Log Files

**LLM Interactions Log** (`llm_interactions.jsonl`):
- Each line is a JSON object
- Contains full LLM request/response
- Includes categorization reasoning

**Error Log** (`categorization_errors.log`):
- Failed categorizations
- Errors with explanations
- Emails that couldn't be processed

---

## Troubleshooting

### Problem: "Connection refused" when running

**Cause:** Ollama is not running

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama (it should auto-start, but if not):
# On macOS, open the Ollama app from Applications
```

### Problem: Model not found

**Cause:** Model not downloaded

**Solution:**
```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

### Problem: "Invalid credentials" or Gmail API error

**Cause:** Gmail authentication issue

**Solution:**
```bash
# Remove existing token
rm token.json

# Run again - it will re-authenticate
gmail-pipeline run --config config_test_3b.yaml
```

### Problem: Too slow / timeout errors

**Solutions:**
1. Use 3B model instead of 7B (faster)
2. Increase timeout in config:
   ```yaml
   timeout: 120  # Increase from 60 or 90
   ```
3. Reduce batch size:
   ```yaml
   batch_size: 10  # Reduce from 20 or 50
   ```

### Problem: Categories are inaccurate

**Solutions:**
1. Try 7B model (more accurate)
2. Edit the `user_prompt` in config to add more specific examples
3. Adjust category descriptions for your specific use case

### Problem: No emails processed

**Possible causes:**
- No emails in inbox matching query
- All emails already processed (check database)

**Solution:**
```bash
# Check Gmail query
# Modify config to use different query:
gmail_query: "in:inbox is:unread"  # Only unread
# Or
gmail_query: "in:inbox after:2024/01/01"  # Date filter
```

---

## Tips & Best Practices

### 1. Always Test First

Run dry-run tests before production:
```bash
# Test first
gmail-pipeline run --config config_test_3b.yaml

# Review results, then run production
gmail-pipeline run --config config_production_3b.yaml
```

### 2. Compare Models

Test both models on the same emails:
```bash
# Run 3B
gmail-pipeline run --config config_test_3b.yaml

# Run 7B (processes SAME emails due to shared database)
gmail-pipeline run --config config_test_7b.yaml

# Compare metrics
diff <(cat ~/.local/share/gmail-llm-labeler/pipeline_metrics_test_3b.json) \
     <(cat ~/.local/share/gmail-llm-labeler/pipeline_metrics_test_7b.json)
```

### 3. Monitor Resource Usage

The 7B model uses more RAM and CPU:
```bash
# Check system resources while running
top -pid $(pgrep ollama)
```

### 4. Incremental Processing

Don't process everything at once:
```bash
# Start small
max_results: 100  # First run

# Then increase
max_results: 1000  # Second run

# Finally full
max_results: 5000  # Full run
```

### 5. Custom Queries

Customize what emails to process:
```yaml
# Only recent emails
gmail_query: "in:inbox after:2024/02/01"

# Exclude already labeled
gmail_query: "in:inbox -label:recruiter -label:automarketing"

# Specific sender
gmail_query: "in:inbox from:example.com"
```

### 6. Backup Before Production

```bash
# Backup Gmail before running production
# Use Gmail Takeout or similar

# Or start with dry-run to verify
dry_run: true
```

---

## Files & Locations

### Configuration Files (Project Directory)
```
/Users/gmarceau/code/gmail-llm-labeler/
├── config_test_3b.yaml          # Test config (3B model)
├── config_test_7b.yaml          # Test config (7B model)
├── config_production_3b.yaml    # Production config (3B model)
├── config_production_7b.yaml    # Production config (7B model)
└── USAGE_GUIDE.md              # This file
```

### Data Files (System Location)
```
~/.local/share/gmail-llm-labeler/
├── email_pipeline.db                     # SQLite database
├── logs/
│   ├── llm_interactions.jsonl           # LLM request/response log
│   └── categorization_errors.log        # Error log
├── pipeline_metrics_test_3b.json        # Metrics (test 3B)
├── pipeline_metrics_test_7b.json        # Metrics (test 7B)
├── pipeline_metrics_production_3b.json  # Metrics (prod 3B)
└── pipeline_metrics_production_7b.json  # Metrics (prod 7B)
```

### Gmail Credentials (Project Directory)
```
/Users/gmarceau/code/gmail-llm-labeler/
├── credentials.json  # OAuth credentials (from Google Cloud)
└── token.json        # Access token (auto-generated)
```

---

## Next Steps

After successful testing:

1. ✅ **Review test results** - Check if categorizations are accurate
2. ✅ **Choose your model** - 3B for speed, 7B for accuracy
3. ✅ **Adjust categories if needed** - Edit the `user_prompt` in config
4. ✅ **Run production** - Start with small batch, then increase
5. ✅ **Monitor Gmail** - Check that labels are applied correctly
6. ✅ **Set up automation** - Consider cron job for regular processing

## Support

- **Project docs:** `/Users/gmarceau/code/gmail-llm-labeler/docs/`
- **GitHub:** https://github.com/ColeMurray/gmail-llm-labeler
- **Ollama docs:** https://ollama.ai/docs

---

**Created:** 2026-02-07
**Models:** Qwen 2.5 3B & 7B via Ollama
**Categories:** recruiter, automarketing, charity_activism, newsletter, action, main
