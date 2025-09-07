from workflow.nodes.check_new_data import check_new_data, new_data_router
from workflow.nodes.check_data_quality import check_data_quality, data_quality_router
from workflow.nodes.update_fs import update_feature_store, is_the_first_day_of_week
from workflow.nodes.check_data_drift import check_data_drift, data_drift_router
from workflow.nodes.check_model_drift import check_model_drift, model_drift_router
from workflow.nodes.retraining import train
from workflow.nodes.init import init_node
from langgraph.graph import StateGraph, START, END
from workflow.schema import State
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

def create_graph():
    builder = StateGraph(State)
    builder.add_node('initialize_pipeline', init_node)
    builder.add_node('validate_new_data', check_new_data)
    builder.add_node('assess_data_quality', check_data_quality)
    builder.add_node('update_feature_group', update_feature_store)
    builder.add_node('detect_data_drift', check_data_drift)
    builder.add_node('detect_model_drift', check_model_drift)
    builder.add_node('retrain_model', train)

    builder.add_conditional_edges('validate_new_data', 
                                  new_data_router,
                                  {
                                      'proceed': 'assess_data_quality',
                                      'terminate': END
                                  })
    builder.add_conditional_edges('assess_data_quality',
                                  data_quality_router,
                                  {
                                      'proceed': 'update_feature_group',
                                      'terminate': END
                                  })
    builder.add_conditional_edges('update_feature_group',
                                  is_the_first_day_of_week,
                                  {
                                      'proceed': 'detect_data_drift',
                                      'terminate': END
                                  })
    builder.add_conditional_edges('detect_data_drift',
                                  data_drift_router,
                                  {
                                      'proceed': 'detect_model_drift',
                                      'trigger_retrain': 'retrain_model'
                                  })
    builder.add_conditional_edges('detect_model_drift',
                                  data_drift_router,
                                  {
                                      'trigger_retrain': 'retrain_model',
                                      'terminate': END
                                  })
    builder.add_edge(START, 'initialize_pipeline')
    builder.add_edge('initialize_pipeline', 'validate_new_data')
    return builder.compile()

if __name__ == '__main__':
    graph = create_graph()
    mermaid_syntax = graph.get_graph().draw_mermaid()
    draw_mermaid_png(mermaid_syntax, output_file_path="my_langgraph_graph.png")
    # config = load_config()
    # graph.invoke({'config': config})
    


