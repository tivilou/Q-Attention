"""Closed plugin registry, explicit scientific imports only."""
def resolve_plugin(plugin_id):
    if plugin_id == 'full_field_encoding':
        import full_field_head
        return full_field_head
    if plugin_id == 'born_context_formation':
        import rmea_context_formation
        return rmea_context_formation
    if plugin_id == 'history_posterior':
        import rmea_history_posterior
        return rmea_history_posterior
    if plugin_id == 'ordered_subsequence':
        import rmea_ordered_subsequence
        return rmea_ordered_subsequence
    if plugin_id == 'phase_presence':
        import rmea_phase_presence
        return rmea_phase_presence
    raise ValueError('Unregistered RMEA technical plugin')
