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
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class KMeansFancy {

	public static void performKMeansFancy(GroupedDataset<String, ListDataset<FImage>, FImage> training, VFSListDataset<FImage> testing){

		/*** Training ***/
		/*Create a DenseSIFT feature extractor with step size of 4 and 8 bins
		 * And use it to create a PyramidDenseSIFT feature extractor with size 4
		 */
		DenseSIFT dsift = new DenseSIFT(4, 8);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4); 
		
		//Get the HardAssigner
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(training, pdsift);
		
		//Create a HomogenousKernelMap and use it to create a feature extractor using the PHOW extractor
		HomogeneousKernelMap kernelMap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, FImage> extractor = kernelMap.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		
		//Develop a set of linear classifiers and train the classifiers on the training data
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(training);

		/*** Testing ***/
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
			if(scores.size() != 0){
				output.put(testing.getFileObject(i).getName().getBaseName(), scores.get(indexGreatestConf).annotation);
			}else{
				output.put(testing.getFileObject(i).getName().getBaseName(), "NoClass");
			}
			
		}

		/*** Write to file ***/
		try{
			Main.writeResults(output, 3);
		}catch (IOException e){
			System.err.println("Unable to write to file, exact error: " + e);
		}

	}
	
	private static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> training, PyramidDenseSIFT<FImage> pdsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		// Gets the PyramidDenseSIFT features for every image in the training set
		for (Entry<String, ListDataset<FImage>> entry : training.entrySet()) 
		{
			for(FImage trainImage : entry.getValue()) 
			{
				pdsift.analyseImage(trainImage.normalise());
				allkeys.add(pdsift.getByteKeypoints(0.005f));
			}
		}
	
		//Keeps only the first 10000 DenseSIFT features
				if (allkeys.size() > 10000)
					allkeys = allkeys.subList(0, 10000);

				//Creates a K-Means classifier with 500 clusters, and clusters the PyramidDenseSIFT features
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500); 
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}
	
	private static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage image) {
	    	//Gets the PyramidDenseSIFT feature of an image
	    	pdsift.analyseImage(image);

	    	//Gets the BOVW from the assigner
	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);
	        
	        //Create a PyramidSpatialAggregator with 8 blocks
	        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(bovw, 2, 4);

	        //Returns the features
	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}
	
}
