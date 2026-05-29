# CSE 151B Project


### GPU Types Used
* NVIDIA A30
* NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition MIG 1g.24gb
* NVIDIA H100 PCIe MIG 1g.20gb

### Approximate Total Inference Time
* Private dataset: 40 hours

## Reproducing Results

### Environment Setup
The packages (with exact versions) used to run this script successfully on an A30 GPU can be found in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### What run_inference() does and how to call it
The `run_inference` function can be found in `submission.py`. This function loads data from `DEFAULT_DATA_PATH`, loads the base model, generates responses, and saves the submission CSV to `DEFAULT_OUTPUT_PATH`.

Modify `DEFAULT_DATA_PATH`, `DEFAULT_OUTPUT_PATH`, and `GPU_ID` at the top of the file as needed.

You can directly call the `run_inference` function or simply run the `submission.py` script with:

```bash
python submission.py
```