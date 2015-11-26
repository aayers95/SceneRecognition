package uk.ac.soton.ecs.ava1g13_se6g13;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class Main {
    public static void main( String[] args ) 
    {
    	try {
			VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("zip:/home/sav/Documents/Soton/COMP3204/CW3/SceneRecognition/src/main/resources/training.zip", ImageUtilities.FIMAGE_READER);
			VFSListDataset<FImage> testing = new VFSListDataset<FImage>("zip:/home/sav/Documents/Soton/COMP3204/CW3/SceneRecognition/src/main/resources/testing.zip", ImageUtilities.FIMAGE_READER);
			
			KNN.performKNN(training, testing);
			
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
    }
    
	// write results to file with the image name and it's prediction 
	public static void writeResults(Map<String, String> results, int runNo) throws IOException 
	{
		File fout = new File("run"+runNo+".txt");
 
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fout)));
	 
		for (Entry<String, String> entry : results.entrySet()) 
		{
			bw.write(entry.getKey() + " " + entry.getValue());
			bw.newLine();
		}
	 
		bw.close();
	}
    
}
