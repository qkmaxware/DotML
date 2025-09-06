using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class FeatureSetTest {

    [TestMethod]
    public void TestDimensions() {
        var batched = new BatchedFeatureSet<double>(new Shape4D(1, 2, 3, 4));
        Assert.AreEqual(4, batched.Rank);
        Assert.AreEqual(1, batched.Batches);
        Assert.AreEqual(1, batched.GetDimension(0));
        Assert.AreEqual(2, batched.Channels);
        Assert.AreEqual(2, batched.GetDimension(1));
        Assert.AreEqual(3, batched.Rows);
        Assert.AreEqual(3, batched.GetDimension(2));
        Assert.AreEqual(4, batched.Columns);
        Assert.AreEqual(4, batched.GetDimension(3));
        Assert.AreEqual(batched.Shape.Count, batched.Size);

        var unbatched = new FeatureSet<double>(new Shape3D(1, 2, 3));
        Assert.AreEqual(3, unbatched.Rank);
        Assert.AreEqual(1, unbatched.Channels);
        Assert.AreEqual(1, unbatched.GetDimension(0));
        Assert.AreEqual(2, unbatched.Rows);
        Assert.AreEqual(2, unbatched.GetDimension(1));
        Assert.AreEqual(3, unbatched.Columns);
        Assert.AreEqual(3, unbatched.GetDimension(2));
        Assert.AreEqual(unbatched.Shape.Count, unbatched.Size);
    }

    [TestMethod]
    public void TestReshape() {
        var batched = BatchedFeatureSet<double>.FromJagged(
            // Batches
            [ 
                // First batch
                [
                    // First feature
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                ]
            ]
        );
        Assert.AreEqual(new Shape4D(1, 1, 3, 3), batched.Shape);

        // Test reshaping to different 4D shapes
        var reshaped4D = batched.Reshape(new Shape4D(1, 1, 1, 9));
        Assert.AreEqual(new Shape4D(1, 1, 1, 9), reshaped4D.Shape);
        Assert.IsTrue(batched.FlattenElements().SequenceEqual(reshaped4D.FlattenElements()));

        // Test reshaping to 3D (unbatched)
        var reshaped3D = batched.Reshape(new Shape3D(1, 3, 3));
        Assert.AreEqual(new Shape3D(1, 3, 3), reshaped3D.Shape);
        Assert.IsTrue(batched.FlattenElements().SequenceEqual(reshaped3D.FlattenElements()));

        // Test reshaping to 2D (unbatched)
        var reshaped2D = batched.Reshape(new Shape2D(3, 3));
        Assert.AreEqual(new Shape2D(3, 3), reshaped3D.Shape);
        Assert.IsTrue(batched.FlattenElements().SequenceEqual(reshaped2D.FlattenRows()));
    }

    [TestMethod]
    public void TestPermute() {
        var batched = BatchedFeatureSet<double>.FromJagged(
            [ 
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                ]
            ]
        );
        Assert.AreEqual(new Shape4D(1, 1, 3, 3), batched.Shape);

        // Test permuting rows and columns (2,3)
        var permuted1 = batched.Permute(0, 1, 3, 2);
        Assert.AreEqual(new Shape4D(1, 1, 3, 3), permuted1.Shape);
        Assert.AreEqual(1, permuted1[0,0,0,0]);
        Assert.AreEqual(4, permuted1[0,0,0,1]); 
        Assert.AreEqual(7, permuted1[0,0,0,2]);
        Assert.AreEqual(2, permuted1[0,0,1,0]);

        // Test permuting channels and rows (1,2)  
        var permuted2 = batched.Permute(0, 2, 1, 3);
        Assert.AreEqual(new Shape4D(1, 3, 1, 3), permuted2.Shape);
        Assert.AreEqual(1, permuted2[0,0,0,0]);
        Assert.AreEqual(4, permuted2[0,1,0,0]);
        Assert.AreEqual(7, permuted2[0,2,0,0]);
        
        // Test permuting batches and channels (0,1)
        var permuted3 = batched.Permute(1, 0, 2, 3); 
        Assert.AreEqual(new Shape4D(1, 1, 3, 3), permuted3.Shape);
        Assert.AreEqual(1, permuted3[0,0,0,0]);
        Assert.AreEqual(2, permuted3[0,0,0,1]);
        Assert.AreEqual(3, permuted3[0,0,0,2]);
    }

}