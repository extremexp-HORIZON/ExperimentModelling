/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.imsi.queryEREngine.apache.calcite.rel.rules;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.plan.ViewExpanders;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableScan;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/**
 * Planner rule that converts a
 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableScan} to the result
 * of calling {@link RelOptTable#toRel}.
 *
 * @deprecated {@code org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories.TableScanFactoryImpl}
 * has called {@link RelOptTable#toRel(RelOptTable.ToRelContext)}.
 */
@Deprecated // to be removed before 2.0
public class TableScanRule extends RelOptRule {
	//~ Static fields/initializers ---------------------------------------------

	public static final TableScanRule INSTANCE =
			new TableScanRule(RelFactories.LOGICAL_BUILDER);

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a TableScanRule.
	 *
	 * @param relBuilderFactory Builder for relational expressions
	 */
	public TableScanRule(RelBuilderFactory relBuilderFactory) {
		super(operand(LogicalTableScan.class, any()), relBuilderFactory, null);
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public void onMatch(RelOptRuleCall call) {
		final LogicalTableScan oldRel = call.rel(0);
		RelNode newRel =
				oldRel.getTable().toRel(
						ViewExpanders.simpleContext(oldRel.getCluster()));
		call.transformTo(newRel);
	}
}
