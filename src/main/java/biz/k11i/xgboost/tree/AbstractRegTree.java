package biz.k11i.xgboost.tree;

import java.io.IOException;
import java.io.Serializable;

import biz.k11i.xgboost.util.FVec;
import biz.k11i.xgboost.util.ModelReader;

/**
 * Provides basic interface and common functionality for a binary decision tree to be loaded and
 * evaluated.
 */
abstract public class AbstractRegTree implements Serializable {
  /**
   * Loads the model from a provided ModelReader
   * @param reader
   * @throws IOException
   */
  public final void loadModel(ModelReader reader) throws IOException {
    loadModel(new Param(reader));
  }

  /**
   * Loads the model from a parameters instance
   * @param param
   */
  public abstract void loadModel(Param param);

  protected int getRootNode() {
    return 0;
  }

  protected abstract int getNextNode(int node, FVec feat);

  protected abstract boolean isLeafNode(int node);

  protected abstract double getLeafValue(int node);

  /**
   * Returns the leaf node index for the given fvec starting at the tree's root
   * @param feat feature vector to evaluate tree on
   * @return leaf node index
   */
  public final int getLeafIndex(FVec feat) {
    return getLeafIndex(feat, getRootNode());
  }

  /**
   * Returns the leaf node index for the given fvec starting at the provided node
   * @param feat feature vector to test tree on
   * @param node first node considered for evaluation
   * @return leaf node index
   */
  public final int getLeafIndex(FVec feat, int node) {
    while (!isLeafNode(node)) {
      node = getNextNode(node, feat);
    }

    return node;
  }

  /**
   * Returns the leaf node value for the given fvec starting at the tree's root
   * @param feat feature vector to evaluate tree on
   * @return leaf node index
   */
  public final double getLeafValue(FVec feat) {
    return getLeafValue(getLeafIndex(feat));
  }

  /**
   * Parameters.
   */
  public static class Param implements Serializable {
    /*! \brief number of start root */
    final int num_roots;
    /*! \brief total number of nodes */
    final int num_nodes;
    /*!\brief number of deleted nodes */
    final int num_deleted;
    /*! \brief maximum depth, this is a statistics of the tree */
    final int max_depth;
    /*! \brief  number of features used for tree construction */
    final int num_feature;
    /*!
     * \brief leaf vector size, used for vector tree
     * used to store more than one dimensional information in tree
     */
    final int size_leaf_vector;
    /*! \brief reserved part */
    final int[] reserved;

    final Node[] nodeInfo;

    public Param(ModelReader reader) throws IOException {
      num_roots = reader.readInt();
      num_nodes = reader.readInt();
      num_deleted = reader.readInt();
      max_depth = reader.readInt();
      num_feature = reader.readInt();

      size_leaf_vector = reader.readInt();
      reserved = reader.readIntArray(31);

      nodeInfo = new Node[num_nodes];

      for (int i = 0; i < num_nodes; i++) {
        nodeInfo[i] = new Node(i, reader);
      }

      for (int i = 0; i < num_nodes; i++) {
        nodeInfo[i].readStats(reader);
      }
    }
  }

  /**
   * Stores attributes of a tree node.
   * Later it is transformed to int[] array.
   */
  public static class Node implements Serializable {
    final int id;
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    final int parent_;
    // pointer to left, right
    public int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    public  /* unsigned */ int sindex_;
    // extra info (leaf_value or split_cond)
    public float leaf_value;
    public float split_cond;

    public int _defaultNext;
    public int _splitIndex;
    public boolean _isLeaf;

    /*! \brief loss change caused by current split */
    float loss_chg;
    /*! \brief sum of hessian values, used to measure coverage of data */
    float sum_hess;
    /*! \brief weight of current node */
    float base_weight;
    /*! \brief number of child that is leaf node known up to now */
    int leaf_child_cnt;

    Node(int id, ModelReader reader) throws IOException {
      this.id = id;

      parent_ = reader.readInt();
      cleft_ = reader.readInt();
      cright_ = reader.readInt();
      sindex_ = reader.readInt();

      if (is_leaf()) {
        leaf_value = reader.readFloat();
        split_cond = Float.NaN;
      } else {
        split_cond = reader.readFloat();
        leaf_value = Float.NaN;
      }

      _defaultNext = cdefault();
      _splitIndex = split_index();
      _isLeaf = is_leaf();
    }

    void readStats(ModelReader reader) throws IOException {
      loss_chg = reader.readFloat();
      sum_hess = reader.readFloat();
      base_weight = reader.readFloat();
      leaf_child_cnt = reader.readInt();
    }

    boolean is_leaf() {
      return cleft_ == -1;
    }

    public int split_index() {
      return (int) (sindex_ & ((1l << 31) - 1l));
    }

    int cdefault() {
      return default_left() ? cleft_ : cright_;
    }

    public boolean default_left() {
      return (sindex_ >>> 31) != 0;
    }

    int next(FVec feat) {
      Number fvalue = feat.fvalue(_splitIndex);
      if (fvalue == null) {
        return _defaultNext;
      }
      return (fvalue.doubleValue() < split_cond) ? cleft_ : cright_;
    }
  }
}
