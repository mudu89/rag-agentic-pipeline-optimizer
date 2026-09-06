import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

counter = 0


class BadProcessFn(beam.DoFn):
    def process(self, element):
        global counter
        counter += 1
        parts = element.split(",")
        key = parts[0]
        val = int(parts[1])
        yield (key, val)


def run():
    options = PipelineOptions()
    with beam.Pipeline(options=options) as p:
        (
            p
            | "Read" >> beam.io.ReadFromText(
                "gs://my-bucket/input.csv"
            )
            | "Process" >> beam.ParDo(BadProcessFn())
           
            | "Group" >> beam.GroupByKey()
            | "Write" >> beam.io.WriteToText(
                "gs://my-bucket/output"
            )
        )


if __name__ == "__main__":
    run()

 # Side effect / non-pure function
# Anti-pattern: Slow, unbundled heavy parsing per line
 # Inefficient single-item yields
        # instead of batching when applicable
         # Anti-pattern: Using raw GroupByKey for a sum
                    # instead of CombineValues/CombineFn