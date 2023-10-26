package me.escoffier.datascience.rhods;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import inference.GRPCInferenceService;
import inference.GrpcPredictV2;
import io.quarkus.grpc.GrpcClient;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


@ApplicationScoped
public class InferenceService {

	@GrpcClient("inference")
	GRPCInferenceService service;

	@Inject
	RecognizedObject objects;

	public List<String> inference(File img) throws IOException {
		// new File("/Users/clement/Downloads/reproducers/rhods-test/IMG_1728.jpg")
		INDArray i = preprocess(img);
		var iterable = new FloatFlattener(i);


		GrpcPredictV2.ModelInferRequest.InferInputTensor images = GrpcPredictV2.ModelInferRequest.InferInputTensor.newBuilder()
				.setName("data")
				.addAllShape(List.of(1L, 3L, 224L, 224L))
				.setDatatype("FP32")
				.setContents(GrpcPredictV2.InferTensorContents.newBuilder()
						.addAllFp32Contents(iterable.get())
						.build())
				.build();

		GrpcPredictV2.ModelInferRequest request = GrpcPredictV2.ModelInferRequest.newBuilder()
				.addInputs(images)
				.setModelName("test")
				.build();

		GrpcPredictV2.ModelInferResponse response = service.modelInfer(request)
				.await().indefinitely();

		return postprocessing(response.getRawOutputContents(0));

	}

	public static INDArray preprocess(File file) throws IOException {
		BufferedImage temp = ImageIO.read(file);
		var image = new Java2DNativeImageLoader(temp.getHeight(), temp.getWidth(), 3).asMatrix(temp, true);

		image = Nd4j.squeeze(image, 0);

		image = image.permute(1, 2, 0);

		image.divi(255.0);

		// Resize (batch first)
		image = image.reshape(1, image.shape()[0], image.shape()[1], image.shape()[2]);
		image = Nd4j.image().imageResize(image, Nd4j.createFromArray(256, 256), ImageResizeMethod.ResizeBilinear);
		// Crop
		image = Nd4j.squeeze(image, 0); // De-batch

		int y0 = (int) Math.floorDiv(image.shape()[0] - 224, 2);
		int x0 = (int) Math.floorDiv(image.shape()[1] - 224, 2);
		image = image.get(
				NDArrayIndex.interval(y0, y0 + 224), // Crop along the height dimension
				NDArrayIndex.interval(x0, x0 + 224), // Crop along the width dimension
				NDArrayIndex.all()              // Keep all channels
		);

		image.subi(Nd4j.createFromArray(0.485f, 0.456f, 0.406f));
		image.divi(Nd4j.createFromArray(0.229f, 0.224f, 0.225f));

		image = image.permute(2, 0, 1); // Channel first

		//image = image.castTo(DataType.FLOAT);
		image = Nd4j.expandDims(image, 0); // Add batch dimension

		image = image.dup().detach();

		return image;
	}

	private List<String> postprocessing(ByteString contents) throws IOException {
		try (var output = Nd4j.create(toFloats(contents))) {
			INDArray[] arrays = Nd4j.sortWithIndices(output.dup().detach(), 0, false);
			int index1 = arrays[0].getInt(0);
			int index2 = arrays[0].getInt(1);
			int index3 = arrays[0].getInt(2);

			float prob1 = arrays[1].getFloat(index1);
			float prob2 = arrays[1].getFloat(index2);
			float prob3 = arrays[1].getFloat(index3);

			List<String> found = new ArrayList<>();
			if (prob1 * 100 > 50) {
				found.add(objects.get(index1));
			}
			if (prob2 * 100 > 50) {
				found.add(objects.get(index2));
			}
			if (prob3 * 100 > 50) {
				found.add(objects.get(index3));
			}

			return found;
		}
	}

	public static float[] toFloats(ByteString output) throws IOException {
		CodedInputStream stream = output.newCodedInput();
		float[] data = new float[output.size()];
		int pos = 0;
		while (!stream.isAtEnd()) {
			data[pos] = stream.readFloat();
			pos++;
		}
		return data;
	}

	public static class FloatFlattener {
		private final List<Float> data;

		public FloatFlattener(INDArray array) {
			if (array.isAttached()) {
				array = array.dup().detach();
			}
			float[] floats = array.data().asFloat();
			data = new ArrayList<>(floats.length);
			for (Float f : floats) {
				data.add(f);
			}
		}

		public List<Float> get() {
			return data;
		}
	}

}
