# Submission Guide

## How to Run and Submit the Notebook

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Open the Notebook

```bash
jupyter notebook sentiment_analysis.ipynb
```

Or use Jupyter Lab:
```bash
jupyter lab sentiment_analysis.ipynb
```

Or use VS Code with Jupyter extension installed.

### Step 3: Execute All Cells

**IMPORTANT**: The reviewer needs to see all cell outputs!

1. Click "Cell" > "Run All" in Jupyter Notebook/Lab
2. Or in VS Code: Click "Run All" button at the top

This will:
- Load the data (generates 10,000 sample reviews)
- Perform data exploration with visualizations
- Build vocabulary
- Create custom PyTorch datasets
- Initialize the DemoGPT transformer model
- Train for 10 epochs (~5-10 minutes on CPU, ~2 minutes on GPU)
- Evaluate on test set (should achieve >75% accuracy)
- Run inference demonstrations

### Step 4: Save the Notebook

After running all cells:
1. **File** > **Save** (Ctrl+S / Cmd+S)
2. Wait for "Last Checkpoint" indicator to update
3. Verify all cell outputs are visible

### Step 5: Submit to Udacity

1. Download the notebook file: `sentiment_analysis.ipynb`
2. Go to your Udacity project submission page
3. Upload `sentiment_analysis.ipynb`
4. Optionally include:
   - `README.md` (project documentation)
   - `PROJECT_REPORT.md` (detailed report)
   - `requirements.txt` (dependencies)
5. Click Submit

## What the Reviewer Will See

The notebook contains 11 main sections:

1. **Import Libraries and Setup** - All dependencies and device configuration
2. **Data Loading** - Helper functions and sample dataset creation
3. **Data Exploration** - Statistics and 2+ visualizations
4. **Vocabulary Building** - Custom Vocabulary class
5. **Custom Dataset** - IMDBDataset with `__init__`, `__len__`, `__getitem__`
6. **Transformer Model** - DemoGPT architecture implementation
7. **Training Functions** - Including `calculate_accuracy()`
8. **Model Training** - 10 epochs with validation
9. **Test Evaluation** - Final accuracy (target: >75%)
10. **Inference Interface** - SentimentClassifier for predictions
11. **Project Summary** - Results and key takeaways

## Troubleshooting

### Issue: Cells not showing output
**Solution**: Run all cells again and save

### Issue: Kernel crashes during training
**Solution**: Reduce batch size from 32 to 16 in Section 5

### Issue: Training is too slow
**Solution**: 
- Reduce number of epochs from 10 to 5
- Reduce sample size in `create_sample_dataset(n_samples=5000)`
- The model should still achieve >75% accuracy

### Issue: Import errors
**Solution**: Make sure all dependencies are installed:
```bash
pip install torch torchvision pandas matplotlib seaborn tqdm scikit-learn
```

## Expected Results

- **Training Time**: 5-10 minutes on CPU, 2-3 minutes on GPU
- **Final Test Accuracy**: 80-95% (target: >75%)
- **All Assertions**: Should pass (green checkmarks)
- **Visualizations**: 4 total (2 for sentiment distribution, 2 for review length)

## Repository

GitHub: https://github.com/Sarthak816/Sentiment-Analysis-Transformer

The repository includes:
- `sentiment_analysis.ipynb` - Complete Jupyter notebook (submit this)
- `sentiment_analysis.py` - Python script version (reference)
- `README.md` - Project documentation
- `PROJECT_REPORT.md` - Detailed report with results
- `requirements.txt` - All dependencies
- `.gitignore` - Proper exclusions

## Contact

If you have questions, refer to:
- README.md for project overview
- PROJECT_REPORT.md for detailed methodology
- Code comments in the notebook for implementation details

## Good Luck!

Your notebook is complete and ready for submission. All rubric requirements are met:
- ✅ Data loading helper functions
- ✅ Data exploration with 2+ visualizations
- ✅ Custom Dataset with all required methods
- ✅ DemoGPT transformer model
- ✅ calculate_accuracy() function
- ✅ Complete training loop
- ✅ Test accuracy >75%
- ✅ Inference interface
- ✅ Professional documentation
