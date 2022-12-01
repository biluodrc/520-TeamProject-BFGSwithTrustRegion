### 1. Environment Description

In this project, we implement codes with `Python 3.7`. To run the project, there are four packages needed, including `random`, `numpy`, `matplotlib`, and `unittest`. 

The first three packages are used for the implementation of BFGS with trust region. The `random` and `unittest` is built-in package for Python, while numpy and matplotlib need to be installed by pip or conda. The command for installing `numpy` and `matplotlib` is:

```bash
pip install numpy

pip install matplotlib
```

### 2. Project Description

`bfgs_with_trust_region.py`----The python file including the implementation of BFGS algorithm and trust region algorithm.

`TargetFunction.py`----The python file including several different kinds of target function and their corresponding first-order derivative.

`test_bfgs_with_trust_region.py`----The python file including several unit tests to test the correctness of the implementation of BFGS algorithm and trust region algorithm.

`mian.py`----The python file aggregating all of three files together, because some online python compilers can only compile one python file.

### 3. Run codes

There are two ways to run this project:

#### 3.1 Run codes online

We use https://replit.com/ to run our codes online.

1. Open this website: https://replit.com/ and **Start creating**.

   ![image-20221201001639915](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201001639915.png)

2. Login or Create an account:

   ![image-20221201001732436](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201001732436.png)

3. Create a new project (the blue button on the left upper corner):

   ![image-20221201001823876](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201001823876.png)

4. Select `Python`:

   ![image-20221201001908263](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201001908263.png)

5. Upload our project `.py` documents:

   ![image-20221201002030154](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201002030154.png)

6. Upload the `main.py`, `bfgs_with_trust_region.py`, `TargetFunction.py`, `test_bfgs_with_trust_region.py`:

7. Run codes

   - Run the `main.py` by **Run** (the green button), and we could see that two tests are passed:

     ![image-20221201002444385](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201002444385.png)

   - Select the **Shell** and run script `python bfgs_with_trust_region.py`

   - ![image-20221201002846106](https://github.com/biluodrc/520-TeamProject-BFGSwithTrustRegion/tree/main/README.assets/image-20221201002846106.png)

#### 3.2 Run codes in local environment

To get better presentation, such as watching the plot of trust region algorithm, running codes in local environment is a better choice.

1. **Download** the Python Installer binaries and **Run** the Executable Installer to install python;

2. Open terminal and use `pip` to install packages:

   ```bash
   pip install numpy
   
   pip install matplotlib
   ```

3. Change directory to run the code:

   ```bash
   python test_bfgs_with_trust_region.py # run the unit test for this project
   python main.py # the same as last statement above
   python bfgs_with_trust_region.py # plot the process of BFGS with Trust Region algorithm
   ```

