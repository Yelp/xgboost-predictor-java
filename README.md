xgboost-predictor-java
======================

Pure Java implementation of [XGBoost](https://github.com/dmlc/xgboost/) predictor for online prediction tasks. This fork has been modified by the Ad Delivery team at Yelp to improve online prediction speeds. Be on the lookout for a few upcoming blog posts about our transition to xgboost!

## Using Predictor in Java

```java
package biz.k11i.xgboost.demo;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;

public class HowToUseXgboostPredictor {
    public static void main(String[] args) throws java.io.IOException {
        // If you want to use faster exp() calculation, uncomment the line below
        // ObjFunction.useFastMathExp(true);

        // Load model and create Predictor
        Predictor predictor = new Predictor(
                new java.io.FileInputStream("/path/to/xgboost-model-file"));

        // Create feature vector from dense representation by array
        double[] denseArray = {0, 0, 32, 0, 0, 16, -8, 0, 0, 0};
        FVec fVecDense = FVec.Transformer.fromArray(
                denseArray,
                true /* treat zero element as N/A */);

        // Create feature vector from sparse representation by map
        FVec fVecSparse = FVec.Transformer.fromMap(
                new java.util.HashMap<Integer, Double>() {{
                    put(2, 32.);
                    put(5, 16.);
                    put(6, -8.);
                }});

        // Predict probability or classification
        double[] prediction = predictor.predict(fVecDense);

        // prediction[0] has
        //    - probability ("binary:logistic")
        //    - class label ("multi:softmax")

        // Predict leaf index of each tree
        int[] leafIndexes = predictor.predictLeaf(fVecDense);

        // leafIndexes[i] has a leaf index of i-th tree
    }
}
```


# Benchmark

Our improvements to cache efficiency and tree structure have resulted in more than doubling the online performance for our use case compared to the [the original project](https://github.com/komiya-atsushi/xgboost-predictor-java). There will be a post on [our engineering blog](https://engineeringblog.yelp.com/) soon about how we achieved this.

# Supported models, objective functions and API

- Models
    - "gblinear"
    - "gbtree"
- Objective functions
    - "binary:logistic"
    - "binary:logitraw"
    - "multi:softmax"
    - "multi:softprob"
    - "reg:linear"
- API
    - Predicts probability or classification
        - `Predictor#predict(FVec)`
    - Outputs margin
        - `Predictor#predict(FVec, true /* output margin */)`
    - Predicts leaf index
        - `Predictor#predictLeaf(FVec)`
