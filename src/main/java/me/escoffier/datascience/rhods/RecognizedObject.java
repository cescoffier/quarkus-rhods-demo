package me.escoffier.datascience.rhods;

import io.smallrye.mutiny.Multi;
import io.vertx.mutiny.core.Vertx;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.util.ArrayList;
import java.util.List;

@ApplicationScoped
public class RecognizedObject {

	@Inject
	Vertx vertx;

	List<String> objects = new ArrayList<>();

	@PostConstruct
	public void init() {
		this.objects.addAll(
				vertx.fileSystem().readFile("synset.txt")
						.onItem().transformToMulti(buffer -> Multi.createFrom().items(buffer.toString("UTF-8").split("\n")))
						.map(line -> line.substring(line.indexOf(" ") + 1).trim()) // Extract category
						.collect().asList()
						.await().indefinitely());
	}

	public String get(int index) {
		return objects.get(index);
	}

}
