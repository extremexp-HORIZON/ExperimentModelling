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
package org.imsi.queryEREngine.apache.calcite.schema.impl;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.imsi.queryEREngine.apache.calcite.DataContext;
import org.imsi.queryEREngine.apache.calcite.adapter.java.AbstractQueryableTable;
import org.apache.calcite.linq4j.AbstractEnumerable;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.linq4j.QueryProvider;
import org.apache.calcite.linq4j.Queryable;
import org.apache.calcite.linq4j.function.Experimental;
import org.apache.calcite.linq4j.tree.Expression;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.prepare.Prepare;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableModify;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableModify;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.schema.ModifiableTable;
import org.imsi.queryEREngine.apache.calcite.schema.ScannableTable;
import org.imsi.queryEREngine.apache.calcite.schema.SchemaPlus;
import org.imsi.queryEREngine.apache.calcite.schema.Schemas;
import org.imsi.queryEREngine.apache.calcite.schema.TransientTable;

/**
 * {@link TransientTable} backed by a Java list. It will be automatically added to the
 * current schema when {@link #scan(DataContext)} method gets called.
 *
 * <p>NOTE: The current API is experimental and subject to change without notice.</p>
 */
@Experimental
public class ListTransientTable extends AbstractQueryableTable
implements TransientTable, ModifiableTable, ScannableTable {
	private static final Type TYPE = Object[].class;
	private final List rows = new ArrayList();
	private final String name;
	private final RelDataType protoRowType;

	public ListTransientTable(String name, RelDataType rowType) {
		super(TYPE);
		this.name = name;
		this.protoRowType = rowType;
	}

	@Override public TableModify toModificationRel(
			RelOptCluster cluster,
			RelOptTable table,
			Prepare.CatalogReader catalogReader,
			RelNode child,
			TableModify.Operation operation,
			List<String> updateColumnList,
			List<RexNode> sourceExpressionList,
			boolean flattened) {
		return LogicalTableModify.create(table, catalogReader, child, operation,
				updateColumnList, sourceExpressionList, flattened);
	}

	@Override public Collection getModifiableCollection() {
		return rows;
	}

	@Override public Enumerable<Object[]> scan(DataContext root) {
		// add the table into the schema, so that it is accessible by any potential operator
		root.getRootSchema().add(name, this);

		final AtomicBoolean cancelFlag = DataContext.Variable.CANCEL_FLAG.get(root);

		return new AbstractEnumerable<Object[]>() {
			@Override
			public Enumerator<Object[]> enumerator() {
				return new Enumerator<Object[]>() {
					private final List list = new ArrayList(rows);
					private int i = -1;

					// TODO cleaner way to handle non-array objects?
					@Override public Object[] current() {
						Object current = list.get(i);
						return current.getClass().isArray()
								? (Object[]) current
										: new Object[]{current};
					}

					@Override public boolean moveNext() {
						if (cancelFlag != null && cancelFlag.get()) {
							return false;
						}

						return ++i < list.size();
					}

					@Override public void reset() {
						i = -1;
					}

					@Override public void close() {
					}
				};
			}
		};
	}

	@Override
	public Expression getExpression(SchemaPlus schema, String tableName,
			Class clazz) {
		return Schemas.tableExpression(schema, elementType, tableName, clazz);
	}

	@Override public <T> Queryable<T> asQueryable(QueryProvider queryProvider,
			SchemaPlus schema, String tableName) {
		return new AbstractTableQueryable<T>(queryProvider, schema, this, tableName) {
			@Override
			public Enumerator<T> enumerator() {
				//noinspection unchecked
				return Linq4j.enumerator(rows);
			}
		};
	}

	@Override public RelDataType getRowType(RelDataTypeFactory typeFactory) {
		return typeFactory.copyType(protoRowType);
	}

	@Override public Type getElementType() {
		return TYPE;
	}
}
