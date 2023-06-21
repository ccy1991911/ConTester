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
Please run the following command to get the EDG.
```python
python EDG_build.py
```
