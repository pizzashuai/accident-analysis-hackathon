# Quick Start: LLM-Powered Accident Analysis

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
pip install boto3
```

### 2. Configure AWS Credentials

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment Variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 3. Request Bedrock Model Access

1. Go to AWS Console → Bedrock → Model Access
2. Request access to "Claude 3.5 Sonnet"
3. Wait for approval (usually instant)

### 4. Run Analysis

```bash
cd backend/src/common/features/postprocess
python main.py --track-ids 7 14
```

That's it! The LLM will analyze the collision and generate a comprehensive report.

## 📋 Example Commands

### Basic Analysis

```bash
python main.py --track-ids 7 14
```

### Analyze Specific Frames

```bash
python main.py --track-ids 7 14 --frame-range 10 50
```

### Save Report to File

```bash
python main.py --track-ids 7 14 --output report.json
```

### Use Different AWS Region

```bash
python main.py --track-ids 7 14 --aws-region us-west-2
```

### Disable LLM (Use Static Analysis)

```bash
python main.py --track-ids 7 14 --no-llm
```

## 🎯 What You'll Get

The LLM generates a professional accident analysis report with:

- **Executive Summary**: Collision or near-miss determination
- **Event Timeline**: Chronological sequence with frame citations
- **Impact Analysis**: Severity indicators and key metrics
- **Data Quality**: Assumptions and limitations
- **Conclusion**: Evidence-based final determination

## 🔧 Troubleshooting

### "Could not connect to Bedrock"

- Check AWS credentials: `aws sts get-caller-identity`
- Verify region supports Bedrock
- Ensure IAM permissions include `bedrock:InvokeModel`

### "Model not found"

- Request model access in AWS Console
- Wait for approval
- Verify model ID matches your region

### "No module named 'boto3'"

```bash
pip install boto3
```

## 💡 Tips

1. **First time?** Start with default settings
2. **Testing?** Use `--no-llm` to avoid AWS costs
3. **Production?** Save reports with `--output`
4. **Custom thresholds?** Adjust `--iou-threshold` and `--distance-threshold`

## 📚 Learn More

- Full documentation: `LLM_AGENT_README.md`
- Implementation details: `IMPLEMENTATION_SUMMARY_LLM.md`
- Test the agent: `python test_llm_agent.py`

## 🎬 Example Output

```
================================================================================
ACCIDENT RECONSTRUCTION ANALYSIS (LLM-POWERED)
================================================================================

Analyzing collision between tracks: [7, 14]
Using AWS Bedrock model: anthropic.claude-3-5-sonnet-20241022-v2:0

[ITERATION 1]
  → Tool: load_detections
    Result: Loaded 186 frames

[ITERATION 2]
  → Tool: compute_pair_metrics
    Result: 45 collision candidates

[ITERATION 3]
  → Tool: trace_impact_window
    Result: Collision DETECTED

✓ Analysis complete

# ACCIDENT ANALYSIS REPORT

## EXECUTIVE SUMMARY
COLLISION DETECTED between vehicles Track 7 and Track 14...

[Full report follows]
```

## 🤝 Need Help?

- Check the documentation files in this directory
- Review the test script: `test_llm_agent.py`
- Examine example configurations in `main.py`
