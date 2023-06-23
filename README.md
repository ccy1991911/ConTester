# Sherlock on Specs: Building LTE Conformance Tests through Automated Reasoning

**spec_info**: The code in this folder is used to pre-process TS 24.301. 

Please enter the folder **code** and run the following command.
```python
sh yitiaolong.sh
```

**condition**: The code in this folder is used to find causal relation from TS 24.301.

Please enter the folder **code** and run the following command.
```python
sh yitiaolong.sh
```
**graph**: The code in this folder is used to train the model and generate the EDG. The training data is in **data** folder.

Please enter the folder **code** and run the following command to train the model.
```python
sh batch_train.sh
````
Please run the following command to get the EDG and do some preparation for DP.
```python
python EDG_build.py
python EDG_DP_0.py
```
The file EDG_DP.py is used to find the chain.

Based on the output of EDG_DP.py, we manually fill the parameters and construct the test procedures. Please see the summary of procedures at: [link](https://sites.google.com/view/contester/homepage/security-requirement/details-of-tests).
