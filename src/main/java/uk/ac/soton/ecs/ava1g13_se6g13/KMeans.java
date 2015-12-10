package uk.ac.soton.ecs.ava1g13_se6g13;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class KMeans {

	public static void performKMeans(GroupedDataset<String, ListDataset<FImage>, FImage> training, VFSListDataset<FImage> testing) {

		/*** Train the KMeans classifier ***/
		
		//Create a DenseSIFT feature extractor with step size of 4 and 8 bins
		DenseSIFT dsift = new DenseSIFT(4, 8);
		
		//Get the HardAssigner
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(training, dsift);
		
		//Create a Feature extractor using DenseSIFT and the HardAssigner
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(dsift, assigner);
		
		//Develop a set of linear classifiers and train the classifiers on the training data
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(training);
		
		
		/*** Test ***/
		Map<String, String> output = new HashMap<String, String>();
		
		//Gets the classification confidence for each class for every image in the test set
		for(int i = 0; i < testing.size(); i++){
			List<ScoredAnnotation<String>> scores = ann.annotate(testing.get(i));
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
			Main.writeResults(output, 2);
		}catch (IOException e){
			System.err.println("Unable to write to file, exact error: " + e);
		}
		
	}

	
	/*Given a training set and a DenseSIFT feature extractor  
	 * For every image in the training set, analyses the image using the DenseSIFT feature extractor
	 * Then clusters the visual words and returns the HardAssigner 
	 */
	private static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> training, DenseSIFT dsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		
		// Gets the DenseSIFT features for every image in the training set
		for (Entry<String, ListDataset<FImage>> entry : training.entrySet()) 
		{
			for(FImage trainImage : entry.getValue()) 
			{
				dsift.analyseImage(trainImage.normalise());
				allkeys.add(dsift.getByteKeypoints(0.005f));
			}
		}
	
		//Keeps only the first 10000 DenseSIFT features
		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		//Creates a K-Means classifier with 500 clusters, and clusters the DenseSIFT features
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}
	
	//Our feature extractor
	private static class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage> {

		DenseSIFT dsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;
		
		public BOVWExtractor(DenseSIFT dsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.dsift = dsift;
	        this.assigner = assigner;
	    }
		
		public DoubleFV extractFeature(FImage image) {
			//Gets the DenseSIFT feature of an image
	        dsift.analyseImage(image);
	        
	        //Gets the BOVW from the assigner
	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        //Create a BlockSpatialAggregator with 8 blocks for both X and Y
	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw, 8, 8);

	        //Returns the features
	        return spatial.aggregate(dsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
		
	}
}
