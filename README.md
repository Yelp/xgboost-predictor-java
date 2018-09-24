xgboost-predictor-java
======================

Pure Java implementation of [XGBoost](https://github.com/dmlc/xgboost/)
predictor for online prediction tasks. This fork has been modified by the Ad
Delivery team at Yelp to improve online prediction speeds.

Check out our Yelp Engineering Blog post for more information about how we use
this library in ([part 1](https://engineeringblog.yelp.com/2018/01/building-a-distributed-ml-pipeline-part1.html))
and what modifications we made to the library to make it cache-friendly and
greatly improve its performance and p50 latencies ([part 2](https://engineeringblog.yelp.com/2018/01/growing-cache-friendly-trees-part2.html)).

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

Our improvements to cache efficiency and tree structure have resulted in more
than doubling the online performance for our use case compared to the [the original project](https://github.com/komiya-atsushi/xgboost-predictor-java).

Check out the [Yelp Engineering Blog post](https://engineeringblog.yelp.com/2018/01/growing-cache-friendly-trees-part2.html)
with the latency benchmarks with the Yelp improvements:

![Latency benchmarks](https://engineeringblog.yelp.com/images/posts/2018-01-12-growing-cache-friendly-trees-part2/mean_latency_reg_tree_prediction.png)

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
