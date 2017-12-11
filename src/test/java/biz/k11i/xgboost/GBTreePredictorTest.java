package biz.k11i.xgboost;

import biz.k11i.xgboost.learner.ObjFunction;
import biz.k11i.xgboost.tree.RegTree;
import biz.k11i.xgboost.util.FVec;

import org.junit.After;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.FromDataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.io.IOException;

@RunWith(Theories.class)
public class GBTreePredictorTest extends PredictorTest {

    private static final String MODEL_TYPE = "gbtree";

    @DataPoints("modelName")
    public static final String[] MODEL_NAMES = {
            "binary-logistic",
            "binary-logitraw",
            "multi-softmax",
            "multi-softprob",
    };

    @DataPoints("version")
    public static final String[] VERSIONS = { "40", "47" };

    @DataPoints
    public static final boolean[] USE_JAFAMA = { true, false };

    @Theory
    public void testPredict(
            @FromDataPoints("modelName") String modelName,
            @FromDataPoints("version") String version,
            boolean useJafama) throws IOException {

        ObjFunction.useFastMathExp(useJafama);
        String path = "model/" + MODEL_TYPE + "/" + modelNameWithVersion(version, modelName) + ".model";
        final Predictor predictor = newPredictor(path);

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat);
            }
        });

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict_ntree", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat, false, 1);
            }
        });

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "margin", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat, true);
            }
        });

        if (modelName.startsWith("binary-")) {
            // test predictSingle()
            verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict", new PredictorFunction<double[]>() {
                @Override
                public double[] predict(FVec feat) {
                    return new double[] {predictor.predictSingle(feat)};
                }
            });
        }
    }

  /**
   * Since we have modified the node layout, the addresses of the
   * nodes would be (original index)*BLOCK_SIZE.
   * To match the test case expectations we divide by the block size.
   * @param leafIndicies Array of leaf indices
   * @return Array of leaf indices divided by block size
   */
    public int [] adjustAddressByBlockSize(int[] leafIndicies){
        for(int i = 0; i < leafIndicies.length; i++) {
            leafIndicies[i] = leafIndicies[i]/ RegTree.BLOCK_SIZE;
        }
        return leafIndicies;
    }

    @After
    public void tearDown() {
        ObjFunction.useFastMathExp(false);
    }
}