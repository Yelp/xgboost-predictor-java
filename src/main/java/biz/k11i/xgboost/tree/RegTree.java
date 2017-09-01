package biz.k11i.xgboost.tree;

import biz.k11i.xgboost.util.FVec;
import biz.k11i.xgboost.util.ModelReader;

import java.io.IOException;
import java.io.Serializable;

/**
 * Regression tree.
 *
 * Memory block model.
 *
 * A node is composed of 3 blocks of ints.
 *
 * Block 1
 * ------------------------------------------
 * | Split condition / Leaf Value (32 bits) |
 * ------------------------------------------
 * Block 2
 * -----------------------------------------------------
 * | Left Child Address (32 bits; 0  iff node is leaf) |
 * -----------------------------------------------------
 * Block 3
 * ------------------------------------------------------
 * | Feature Index (31 bits) | Default is right (1-bit) |
 * ------------------------------------------------------
 *
 * Design Limitations:
 * - A complete tree of height has 2^n -1 nodes. Since we allocate 32 bits for a child address, the
 * height cannot exceed 32 for a complete tree.
 * - Feature index is allocated 31 bits. This implementation can only support 2^31 features.
 */
public class RegTree implements Serializable {
  private Param param;
  private int[] nodes;
  private RTreeNodeStat[] stats;
  public static final int BLOCK_SIZE = 3;

  /**
   * Loads model from stream.
   *
   * @param reader input stream
   * @throws IOException If an I/O error occurs
   */
  public void loadModel(ModelReader reader) throws IOException {
    param = new Param(reader);

    nodes = new int[BLOCK_SIZE * param.num_nodes];
    for (int i = 0; i < BLOCK_SIZE * param.num_nodes; i += BLOCK_SIZE) {
      Node node = new Node(reader);
      /*
       * Store node attributes in contiguous memory. Use Bit masks to store and read attributes.
       */
      nodes[i] = createNodeValue(node);
      nodes[i + 1] = createNodeChildren(node);
      nodes[i + 2] = createNodeDefaultAndValue(node);
    }

    stats = new RTreeNodeStat[param.num_nodes];
    for (int i = 0; i < param.num_nodes; i++) {
      stats[i] = new RTreeNodeStat(reader);
    }
  }

  public int createNodeValue(Node nodeObj) {
    if (nodeObj._isLeaf) {
      return Float.floatToRawIntBits(nodeObj.leaf_value);
    }
    return Float.floatToRawIntBits(nodeObj.split_cond);
  }

  public int createNodeChildren(Node nodeObj) throws IOException {
    if (nodeObj.is_leaf()) {
      return 0;
    } else {
      return nodeObj.cleft_ * BLOCK_SIZE;
    }
  }


  public int createNodeDefaultAndValue(Node nodeObj) {
    return (nodeObj.split_index() << 1) | (nodeObj.default_left() ? 0 : 1);
  }


  public int getNextNode(int index, FVec feat) {
    Double fvalue = feat.fvalue(getFeatureIndex(nodes[index + 2]));
    int child = nodes[index + 1];

    if (fvalue == null) {
      return child + BLOCK_SIZE * (nodes[index + 2] & 1);
    }

    if (fvalue < Float.intBitsToFloat(nodes[index])) {
      return child;
    } else {
      return child + BLOCK_SIZE;
    }
  }

  public static final int getLeftChild(int node) {
    return node;
  }

  public static final int getRightChild(int node) {
    return node + BLOCK_SIZE;
  }

  public static final int getFeatureIndex(int node) {
    return node >>> 1;
  }

  public static final boolean isDefaultLeft(int node) {
    return (node & 1) == 0;
  }

  public static final boolean isNotLeaf(int node) {
    return node != 0;
  }

  /**
   * Retrieves nodes from root to leaf and returns leaf index.
   *
   * @param feat    feature vector
   * @param root_id starting root index
   * @return leaf index
   */
  public int getLeafIndex(FVec feat, int root_id) {
    int pid = root_id;
    // Loop till leaf node is reached.
    while (isNotLeaf(nodes[pid + 1])) {
      pid = getNextNode(pid, feat);
    }

    return pid;
  }

  /**
   * Retrieves nodes from root to leaf and returns leaf value.
   *
   * @param feat    feature vector
   * @param root_id starting root index
   * @return leaf value
   */
  public double getLeafValue(FVec feat, int root_id) {
    // Loop till leaf node is reached.
    while (isNotLeaf(nodes[root_id + 1])) {
      root_id = getNextNode(root_id, feat);
    }

    return Float.intBitsToFloat(nodes[root_id]);
  }

  /**
   * Parameters.
   */
  static class Param implements Serializable {
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

    Param(ModelReader reader) throws IOException {
      num_roots = reader.readInt();
      num_nodes = reader.readInt();
      num_deleted = reader.readInt();
      max_depth = reader.readInt();
      num_feature = reader.readInt();

      size_leaf_vector = reader.readInt();
      reserved = reader.readIntArray(31);
    }
  }

  /**
   * Stores attributes of a tree node.
   * Later it is transformed to int[] array.
   */
  public static class Node implements Serializable {
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

    // set parent
    Node(ModelReader reader) throws IOException {
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
      Double fvalue = feat.fvalue(_splitIndex);
      if (fvalue == null) {
        return _defaultNext;
      }
      return (fvalue < split_cond) ? cleft_ : cright_;
    }
  }

  /**
   * Statistics each node in tree.
   */
  static class RTreeNodeStat implements Serializable {
    /*! \brief loss chg caused by current split */
    final float loss_chg;
    /*! \brief sum of hessian values, used to measure coverage of data */
    final float sum_hess;
    /*! \brief weight of current node */
    final float base_weight;
    /*! \brief number of child that is leaf node known up to now */
    final int leaf_child_cnt;

    RTreeNodeStat(ModelReader reader) throws IOException {
      loss_chg = reader.readFloat();
      sum_hess = reader.readFloat();
      base_weight = reader.readFloat();
      leaf_child_cnt = reader.readInt();
    }
  }
}
