package biz.k11i.xgboost;
import biz.k11i.xgboost.tree.RegTree;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class RegTreeTest {
  RegTree regTree = new RegTree();

  @Test(expected = IOException.class)
  public void treeExceedsMaxLength() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node.cright_ = 0x1ffff;
    regTree.createNodeChildren(node);
  }

  @Test
  public void createNodeChildrenTest() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node.cright_ = 0x11;
    node.cleft_ = 0x12;
    int children = regTree.createNodeChildren(node);
    assertEquals((children >>> 16) & RegTree.QUAD_WORD, node.cleft_);
    assertEquals(children & RegTree.QUAD_WORD, node.cright_);
  }

  @Test
  public void createNodeLeafDefaultAndValueLeafTest() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node._isLeaf = true;
    when(node.split_index()).thenReturn(0x5);
    int nodeValue = regTree.createNodeLeafDefaultAndValue(node);
    assertEquals(nodeValue & RegTree.LEAF_MASK, RegTree.LEAF_MASK);
    assertEquals(nodeValue & RegTree.SPLIT_MASK, 0x5);
  }

  @Test
  public void createNodeLeafDefaultAndValueNonLeafTest() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node._isLeaf = false;
    when(node.default_left()).thenReturn(true);
    int nodeValue = regTree.createNodeLeafDefaultAndValue(node);
    assertEquals(nodeValue & RegTree.DEFAULT_MASK, RegTree.DEFAULT_MASK);
  }

  @Test
  public void createNodeLeafDefaultAndValueNonLeafTest() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node._isLeaf = false;
    when(node.default_left()).thenReturn(true);
    int nodeValue = regTree.createNodeLeafDefaultAndValue(node);
    assertEquals(nodeValue & RegTree.DEFAULT_MASK, RegTree.DEFAULT_MASK);
  }
}
