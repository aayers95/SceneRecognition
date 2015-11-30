package uk.ac.soton.ecs.ava1g13_se6g13;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class KMeans {

	public static void performKMeans(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {

		DenseSIFT dsift = new DenseSIFT(4, 4);
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(training, dsift);
		
		System.out.println("Hi");
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(dsift, assigner);
		System.out.println("Hi");
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
				extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		System.out.println("Hi");



	}

	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset<FImage> training, DenseSIFT dsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		for (Entry<String, VFSListDataset<FImage>> entry : training.entrySet()) 
		{
			for(FImage trainImage : entry.getValue()) 
			{
				System.out.println(entry.getKey());
				dsift.analyseImage(trainImage);
				allkeys.add(dsift.getByteKeypoints(0.005f));
			}
		}
	
		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		System.out.println("Hi");
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
		System.out.println("Hi");
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		System.out.println("Hi");
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}
	
	static class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage> {

		DenseSIFT dsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;
		
		public BOVWExtractor(DenseSIFT dsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.dsift = dsift;
	        this.assigner = assigner;
	    }
		
		public DoubleFV extractFeature(FImage object) {
			FImage image = object.getImage();
	        dsift.analyseImage(image);
	        
	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 8, 8);

	        return spatial.aggregate(dsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
		
	}
}
