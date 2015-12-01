package uk.ac.soton.ecs.ava1g13_se6g13;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
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
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.svm.SVMAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

public class SVM {

	public static void performSVM(GroupedDataset<String, ListDataset<FImage>, FImage> training, VFSListDataset<FImage> testing){
		
		/*** Training ***/
		DenseSIFT dsift = new DenseSIFT(4, 8);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4, 6, 8, 10);
		
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(training, pdsift);
		HomogeneousKernelMap kernelMap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = kernelMap.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		
		SVMAnnotator<FImage, String> svm = new SVMAnnotator<FImage, String>(extractor);
		svm.train(training);

		/*** Testing ***/
		Map<String, String> output = new TreeMap<String, String>();

		//Gets the classification confidence for each class for every image in the test set
		for(int i = 0; i < testing.size(); i++){
			List<ScoredAnnotation<String>> scores = svm.annotate(testing.get(i));
			int indexGreatestConf = 0;
			//Gets the annotation which has been given the greatest confidence
			for(int j = 1; j < scores.size(); j++){
				if(scores.get(j).confidence > scores.get(indexGreatestConf).confidence){
					indexGreatestConf = j;
				}
			}
			output.put(testing.getFileObject(i).getName().getBaseName(), scores.get(indexGreatestConf).annotation);
		}

		/*** Write to file ***/
		try{
			Main.writeResults(output, 3);
		}catch (IOException e){
			System.err.println("Unable to write to file, exact error: " + e);
		}

	}
	
	private static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> training, PyramidDenseSIFT<FImage> dsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		for (Entry<String, ListDataset<FImage>> entry : training.entrySet()) 
		{
			for(FImage trainImage : entry.getValue()) 
			{
				dsift.analyseImage(trainImage.normalise());
				allkeys.add(dsift.getByteKeypoints(0.005f));
			}
		}
	
		
		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}
	
	//Our feature extractor
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage image) {
	        pdsift.analyseImage(image);

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 2);

	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}
	
}
