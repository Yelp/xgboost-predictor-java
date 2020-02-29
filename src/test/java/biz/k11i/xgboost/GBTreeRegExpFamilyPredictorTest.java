package biz.k11i.xgboost;

import biz.k11i.xgboost.learner.ObjFunction;

import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.FromDataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.io.IOException;

@RunWith(Theories.class)
public class GBTreeRegExpFamilyPredictorTest extends GBTreePredictorTest {

  private String MODEL_DATA_PATH = "model/agaricus_new.txt.test";

  protected String getTestDataPath() {
    return MODEL_DATA_PATH;
  }

  @DataPoints("modelName")
  // tweedie12: tweedie regresion trained with "tweedie_variance_power": 1.2
  // https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tweedie-regression-objective-reg-tweedie
  public static final String[] MODEL_NAMES = {"poisson", "tweedie12"};

  @DataPoints("version")
  public static final String[] VERSIONS = {"80"};

  @Theory
  public void testPredict(
      @FromDataPoints("modelName") String modelName,
      @FromDataPoints("version") String version,
      boolean useJafama
  ) throws IOException {
    ObjFunction.useFastMathExp(useJafama);
    String path = "model/" + MODEL_TYPE + "/" + modelNameWithVersion(version, modelName) + ".model";
    final Predictor predictor = newPredictor(path);

    verifyDouble(
        MODEL_TYPE,
        modelNameWithVersion(version, modelName),
        "predict",
        predictor::predict
    );

    verifyDouble(
        MODEL_TYPE,
        modelNameWithVersion(version, modelName),
        "predict",
        feat -> new double[]{predictor.predictSingle(feat)}
    );
  }
}
