package uk.ac.soton.ecs.ava1g13_se6g13;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class Main {
    public static void main( String[] args ) 
    {
    	try {
			VFSGroupDataset<FImage> training = 
					new VFSGroupDataset<FImage>( "zip:C:/Users/Aaron/Documents/OpenImajCoursework2/OpenIMAJ-SceneRecognition/src/main/resources/training.zip", ImageUtilities.FIMAGE_READER);
			VFSListDataset<FImage> testing = 
					new VFSListDataset<FImage>( "zip:C:/Users/Aaron/Documents/OpenImajCoursework2/OpenIMAJ-SceneRecognition/src/main/resources/testing.zip", ImageUtilities.FIMAGE_READER);
			System.out.println(training.size());
			System.out.println(testing.size());
			
			
			KNN knn = new KNN(training, testing);
			
		} catch (FileSystemException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
