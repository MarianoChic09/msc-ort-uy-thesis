from typing import Literal
from pydantic import BaseModel
from typing import List, Dict

# Define a strict schema
default_entities = ["BORROWER", "CONTRACTOR", "REQUIREMENT"]
default_relations = ["MUST_COMPLY_WITH", "HAS_OBLIGATION_TO"]

default_validation_schema = {
    "BORROWER": ["MUST_COMPLY_WITH"],
    "CONTRACTOR": ["HAS_OBLIGATION_TO"],
    "REQUIREMENT": [],
}


class Query(BaseModel):
    query: str


# Define a Pydantic model for the request body
class SchemaExtractorRequest(BaseModel):
    entities: List[str]
    relations: List[str]
    validation_schema: Dict[str, List[str]]


def get_default_schema():
    return SchemaExtractorRequest(
        entities=default_entities,
        relations=default_relations,
        validation_schema=default_validation_schema,
    )


# entities = Literal["DOCUMENT", "SECTION", "INTRODUCTION", "OBJECTIVES", "SCOPE_OF_APPLICATION", "REQUIREMENTS", "SUBREQUIREMENTS"]
# relations = Literal["HAS_SECTION", "HAS_INTRODUCTION", "HAS_OBJECTIVES", "HAS_SCOPE_OF_APPLICATION", "HAS_REQUIREMENTS", "HAS_SUBREQUIREMENTS", "NEXT"]

# validation_schema = {
#     "DOCUMENT": ["HAS_SECTION", "NEXT"],
#     "SECTION": ["HAS_INTRODUCTION", "HAS_OBJECTIVES", "HAS_SCOPE_OF_APPLICATION", "HAS_REQUIREMENTS", "NEXT"],
#     "INTRODUCTION": ["NEXT"],
#     "OBJECTIVES": ["NEXT"],
#     "SCOPE_OF_APPLICATION": ["NEXT"],
#     "REQUIREMENTS": ["HAS_SUBREQUIREMENTS", "NEXT"],
#     "SUBREQUIREMENTS": ["NEXT"],
# }
