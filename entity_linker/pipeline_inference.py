from utils.config_utilities import load_config
from entity_linking_pipeline import EntityLinkingPipeline
from entity_linker.entity_linking_pipeline.query_creator import NGramQueryCreator
from entity_linker.entity_linking_pipeline.entity_linking_pipeline import EntityLinkingPipeline
from entity_linker.entity_linking_pipeline.candidates_generator import StringMatchCandidatesGenerator
from entity_linker.entity_linking_pipeline.candidates_ranger import CosineSimRanger, CosineSimRangerWeights


if __name__ == '__main__':
    config = load_config()
    linker = EntityLinkingPipeline(
            NGramQueryCreator(),
            StringMatchCandidatesGenerator(config, is_use_predefined_candidates=True),
            CosineSimRanger()
        )
    term = 'язык программирования Python'
    context = ['язык программирования Python', 'использовался', 'в']
    print(linker.run(term, context))