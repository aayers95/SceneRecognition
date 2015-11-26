package uk.ac.soton.ecs.ava1g13_se6g13;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;

public class KNN 
{
	private VFSGroupDataset<FImage> training;
	private VFSListDataset<FImage> testing;
	private int tinyImageSize = 16;
	private int K = 3;
	private FloatNearestNeighbours knn;
	private String[] sets;
	
	public KNN(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing)
	{
		this.training = training;
		this.testing = testing; 
		train();
		test();
	}

	public void train()
	{
		float[][] tinyFeatures = new float[training.numInstances()][];
		sets = new String[training.numInstances()];
		int index = 0;
		
		for (Entry<String, VFSListDataset<FImage>> entry : training.entrySet()) 
		{
			for(FImage trainImage : entry.getValue()) 
			{
				tinyFeatures[index] = tinyImage(trainImage);
				sets[index] = entry.getKey();
				index++;
			}
		}
		
		knn = new FloatNearestNeighboursExact(tinyFeatures);

	}

	public void test()
	{
		Map<String, String> output = new HashMap<String, String>();
		for(int i = 0; i < testing.size(); i++){
			output.put(testing.getFileObject(i).getName().getBaseName(), sets[findMost(knn.searchKNN(tinyImage(testing.get(i)), K))]);
		}
	}

	public int findMost(List<IntFloatPair> distances) {

	    if (distances == null || distances.size() == 0)
	        return -1; //If code is working this will never happen

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
	
	private float[] tinyImage(FImage image)
	{
		int length = Math.min(image.height, image.width);
		
		FImage extraction = image.extractCenter(length, length);
		
		return ResizeProcessor.resample(extraction, tinyImageSize, tinyImageSize).getFloatPixelVector();		
	}

}
