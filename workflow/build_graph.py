from workflow.nodes.check_new_data import check_new_data, new_data_router
from workflow.nodes.check_data_quality import check_data_quality, data_quality_router
from workflow.nodes.update_fs import update_feature_store, is_the_first_day
from workflow.nodes.init import init_node
from langgraph.graph import StateGraph, START, END
from workflow.schema import State
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from pipeline.utils import load_config

def create_graph():
    builder = StateGraph(State)
    builder.add_node('init_state', init_node)
    builder.add_node('check_new_data', check_new_data)
    builder.add_node('check_data_quality', check_data_quality)
    builder.add_node('update_feature_store', update_feature_store)

    builder.add_conditional_edges('check_new_data', 
                                  new_data_router,
                                  {
                                      'continue': 'check_data_quality',
                                      'END': END
                                  })
    builder.add_conditional_edges('check_data_quality',
                                  data_quality_router,
                                  {
                                      'continue': 'update_feature_store',
                                      'END': END
                                  })
    # builder.add_conditional_edges('update_feature_store',
    #                               is_the_first_day,
    #                               {
    #                                   'continue': 'check_data_drift',
    #                                   'END': END
    #                               })
    builder.add_edge(START,'init_state')
    builder.add_edge('init_state', 'check_new_data')
    builder.add_edge('update_feature_store',END)
    return builder.compile()

if __name__ == '__main__':
    graph = create_graph()
    # mermaid_syntax = graph.get_graph().draw_mermaid()
    # draw_mermaid_png(mermaid_syntax, output_file_path="my_langgraph_graph.png", max_retries=5)
    config = load_config()
    graph.invoke({'config': config})
    


