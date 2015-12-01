package uk.ac.soton.ecs.ava1g13_se6g13;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;

public class KNN 
{
	private static int tinyImageSize = 16;
	
	public static void performKNN(GroupedDataset<String, ListDataset<FImage>, FImage> training, VFSListDataset<FImage> testing){
		
		/*** Train the KNN classifier ***/
		float[][] tinyFeatures = new float[training.numInstances()][];
		String[] sets = new String[training.numInstances()];
		int index = 0;
		
		// Gets the tiny image feature for every image within every classification
		for (Entry<String, ListDataset<FImage>> entry : training.entrySet()) 
		{
			for(FImage trainImage : entry.getValue()) 
			{
				tinyFeatures[index] = tinyImage(trainImage);
				sets[index] = entry.getKey();
				index++;
			}
		}
		
		//Trains the KNN classifier with the tiny features
		FloatNearestNeighboursExact knn = new FloatNearestNeighboursExact(tinyFeatures);
	
		/*** Test ***/
		Map<String, String> output = new TreeMap<String, String>();
		
		//Gets the classification for every image in the test set
		//K is set to the sqrt of the number of instances in the training set
		for(int i = 0; i < testing.size(); i++){
			output.put(testing.getFileObject(i).getName().getBaseName(), sets[findMost(knn.searchKNN(tinyImage(testing.get(i)), (int) Math.floor(Math.sqrt(training.numInstances()))))]);
		}
	
		/*** Write to file ***/
		try{
			Main.writeResults(output, 1);
		}catch (IOException e){
			System.err.println("Unable to write to file, exact error: " + e);
		}
		
	}
	
	// Returns the tiny image feature vector
	private static float[] tinyImage(FImage image){
		int length = Math.min(image.height, image.width);
		FImage extraction = image.extractCenter(length, length);
		return ResizeProcessor.resample(extraction, tinyImageSize, tinyImageSize).getFloatPixelVector();		
	}
	
	// Returns the index of the most number of occurrences in the list of distances
	private static int findMost(List<IntFloatPair> distances) {

	    int previous = distances.get(0).first;
	    int popular = distances.get(0).first;
	    int count = 1;
	    int maxCount = 1;

	    for (int i = 1; i < distances.size(); i++) {
	        if (distances.get(i).first == previous)
	            count++;
	        else {
	            if (count > maxCount) {
	                popular = distances.get(i-1).first;
	                maxCount = count;
	            }
	            previous = distances.get(i).first;
	            count = 1;
	        }
	    }

	    return count > maxCount ? distances.get(distances.size()-1).first : popular;
	}

}
