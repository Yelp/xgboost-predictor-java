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
 * A node is composed of 4 blocks of ints.
 *
 *  Block 1
 *  ________________________________________
 * | Split condition / Leaf Value (32 bits) |
 *  ----------------------------------------
 *  Block 2
 *  _____________________________________________________________
 * | Left Child Address (16-bits) | Right Child Address (16-bits)|
 *  -------------------------------------------------------------
 *  Block 3
 *  _________________________
 * | Feature Index (32 bits) |
 *  -------------------------
 *  Block 4
 *  ___________________________________________________
 * | Is Leaf (1-bit) | Default (left or right) (1-bit) |
 *  ---------------------------------------------------
 *
 *
 *
 */
public class RegTree implements Serializable {
  private Param param;
  private int[] nodes;
  private RTreeNodeStat[] stats;
  private final int BLOCK_SIZE = 4;

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
      /**
       * Store node attributes in contiguous memory. Use Bit masks to store and read attributes.
       */
      nodes[i] = createNodeValue(node);
      nodes[i + 1] = createNodeChildren(node);
      nodes[i + 2] = node.split_index();
      nodes[i + 3] = createNodeLeafDefault(node);
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

  public int createNodeChildren(Node nodeObj) {
    int children = (nodeObj.cright_ & 0xffff);
    children = children | ((nodeObj.cleft_ & 0xffff) << 16);
    return children;
  }

  public int createNodeLeafDefault(Node nodeObj) {
    if (nodeObj._isLeaf) {
      return 2; //Binary code 10
    }
    if (nodeObj.default_left()) {
      return 1; // Binary code 01
    }
    return 0; //Binary code 00. Impossible to be a leaf node and have a default. 11 impossible.
  }


  public int getNextNode(int index, FVec feat) {
    double fvalue = feat.fvalue(nodes[index + 2]);
    if (fvalue != fvalue) {  // is NaN?
      if (nodes[index + 3] == 1) {
        return (((nodes[index + 1] >>> 16) & 0xffff) <<2);
      }
      return ((nodes[index + 1] & 0xffff) << 2);
    }
    // Since the the node is of size 4, we multiply the address by 4 by left shifting by 2.
    return (fvalue < Float.intBitsToFloat(nodes[index])) ?
        (((nodes[index + 1] >>> 16) & 0xffff) << 2)
        : ((nodes[index + 1] & 0xffff) << 2);
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
    while (nodes[pid  + 3] != 2 ) {
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
    while (nodes[root_id  + 3] != 2) {
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

  static class Node implements Serializable {
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    // pointer to left, right
    public final int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    public final /* unsigned */ int sindex_;
    // extra info (leaf_value or split_cond)
    public final float leaf_value;
    public final float split_cond;

    public final int _defaultNext;
    public final int _splitIndex;
    public final boolean _isLeaf;

    // set parent
    Node(ModelReader reader) throws IOException {
      reader.readInt();
      cleft_ = reader.readInt();
      cright_ = reader.readInt();
      sindex_ = reader.readInt();
      _isLeaf = (cleft_ == -1);

      if (_isLeaf) {
        leaf_value = reader.readFloat();
        split_cond = Float.NaN;
      } else {
        split_cond = reader.readFloat();
        leaf_value = Float.NaN;
      }

      _defaultNext = cdefault();
      _splitIndex = split_index();
    }

    int split_index() {
      return (int) (sindex_ & ((1l << 31) - 1l));
    }

    int cdefault() {
      return default_left() ? cleft_ : cright_;
    }

    boolean default_left() {
      return (sindex_ >>> 31) != 0;
    }

    int next(FVec feat) {
      double fvalue = feat.fvalue(_splitIndex);
      if (fvalue != fvalue) {  // is NaN?
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
